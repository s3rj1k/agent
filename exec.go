// SPDX-License-Identifier: Unlicense

package main

import (
	"bytes"
	"cmp"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/joakimcarlsson/ai/tool"
	"github.com/joakimcarlsson/ai/tool/functiontool"
)

const (
	defaultCommandTimeout    = duration(2 * time.Minute)
	defaultMaxCommandTimeout = duration(15 * time.Minute)
	defaultShellProgram      = "/bin/bash"
)

//go:embed config/prompts/exec-tool.txt
var execToolDescription string

//go:embed config/shell.yaml.tmpl
var configShell string

type ShellConfig struct {
	Program           string   `yaml:"program"`
	Dir               string   `yaml:"dir"`
	CommandTimeout    duration `yaml:"command_timeout"`
	MaxCommandTimeout duration `yaml:"max_command_timeout"`
}

type execParams struct {
	Env            map[string]string `json:"env,omitempty" desc:"Extra environment variables for this command, as an object of name to string value. Added to the inherited environment rather than replacing it."`
	TimeoutSeconds *int              `json:"timeout_seconds,omitempty" desc:"Optional per-command timeout in seconds. Omit to use the default. Values above the configured maximum are clamped."`
	Command        string            `json:"command" desc:"The program text to run, passed to the interpreter as one argument. Required; must be a non-empty string."`
	Program        string            `json:"program,omitempty" desc:"Interpreter to run the command with. Defaults to the configured shell. Set it to python3, node, perl or anything else on PATH to write in that language directly rather than quoting it through the shell."`
	ProgramFlag    string            `json:"program_flag,omitempty" desc:"Flag the interpreter takes its program under. Defaults to -c, which suits sh, bash and python. Use -e for node, perl and ruby, where -c means check rather than run."`
	Dir            string            `json:"dir,omitempty" desc:"Working directory for this command. A relative path resolves against the agent working directory. Defaults to it when unset."`
	OutputFile     string            `json:"output_file,omitempty" desc:"Write standard output to this file instead of returning it. Use it when output would be large. A relative path resolves against dir. Standard error is still returned, and the response reports the byte count rather than the content."`
	Stdin          string            `json:"stdin,omitempty" desc:"Optional data piped to the command's standard input. Use this rather than a heredoc when writing file contents."`
	Append         bool              `json:"append,omitempty" desc:"Append to output_file instead of truncating it."`
	Mkdir          bool              `json:"mkdir,omitempty" desc:"Create the parent directory of output_file if it does not exist."`
}

type execTool struct {
	echo       io.Writer
	shell      string
	dir        string
	timeout    time.Duration
	maxTimeout time.Duration
}

func shellSection() (string, error) {
	return renderConfig("shell", configShell, nil)
}

func normalizeShell(cfg *Config, path string) error {
	cfg.Shell.Program = cmp.Or(cfg.Shell.Program, defaultShellProgram)

	if cfg.Shell.CommandTimeout <= 0 {
		cfg.Shell.CommandTimeout = defaultCommandTimeout
	}

	if cfg.Shell.MaxCommandTimeout <= 0 {
		cfg.Shell.MaxCommandTimeout = defaultMaxCommandTimeout
	}

	cfg.Shell.MaxCommandTimeout = max(cfg.Shell.MaxCommandTimeout,
		cfg.Shell.CommandTimeout)

	if _, err := exec.LookPath(cfg.Shell.Program); err != nil {
		return fmt.Errorf("%s: shell.program %q: %w",
			path, cfg.Shell.Program, err)
	}

	if cfg.Shell.Dir != "" {
		if fi, err := os.Stat(cfg.Shell.Dir); err != nil || !fi.IsDir() {
			return fmt.Errorf("%s: shell.dir %q is not a directory",
				path, cfg.Shell.Dir)
		}
	}

	return nil
}

func (t *execTool) openOutput(
	p execParams,
	dir string,
) (*os.File, string, int64, error) {
	path := p.OutputFile
	if !filepath.IsAbs(path) {
		path = filepath.Join(dir, path)
	}

	if p.Mkdir {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, "", 0, fmt.Errorf(
				"exec: creating directory for %q: %w", path, err)
		}
	}

	flags := os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	if p.Append {
		flags = os.O_CREATE | os.O_WRONLY | os.O_APPEND
	}

	f, err := os.OpenFile(path, flags, 0o644)
	if err != nil {
		return nil, "", 0, fmt.Errorf("exec: opening %q: %w", path, err)
	}

	var before int64

	if p.Append {
		if fi, statErr := f.Stat(); statErr == nil {
			before = fi.Size()
		}
	}

	return f, path, before, nil
}

func (t *execTool) run(ctx context.Context, p execParams) (tool.Response, error) {
	command := strings.TrimSpace(p.Command)
	if command == "" {
		return tool.NewTextErrorResponse(
			`exec: "command" is required and must be a non-empty string.`,
		), nil
	}

	program, flag := cmp.Or(p.Program, t.shell), cmp.Or(p.ProgramFlag, "-c")
	if _, err := exec.LookPath(program); err != nil {
		return tool.NewTextErrorResponse(fmt.Sprintf(
			"exec: program %q is not on PATH: %v", program, err)), nil
	}

	timeout := t.timeout
	if p.TimeoutSeconds != nil && *p.TimeoutSeconds > 0 {
		secs := min(*p.TimeoutSeconds, int(t.maxTimeout/time.Second)+1)
		timeout = min(time.Duration(secs)*time.Second, t.maxTimeout)
	}

	banner := command
	if p.Program != "" {
		banner = program + " " + flag + "\n" + command
	}

	_, _ = fmt.Fprintf(t.echo, "$ %s\n", oneBlock(banner))

	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	dir := t.dir
	if p.Dir != "" {
		dir = p.Dir
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(t.dir, dir)
		}
	}

	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		return tool.NewTextErrorResponse(fmt.Sprintf(
			"exec: dir %q is not a directory", dir)), nil
	}

	var out, errOut bytes.Buffer

	stdout := io.Writer(&out)
	written := int64(0)

	before := int64(0)

	outPath := p.OutputFile
	if outPath != "" {
		f, path, size, err := t.openOutput(p, dir)
		if err != nil {
			return tool.NewTextErrorResponse(err.Error()), nil
		}

		defer func() { _ = f.Close() }()

		outPath, stdout, before = path, f, size
	}

	cmd := exec.CommandContext(runCtx, program, flag, command)
	cmd.Dir = dir

	if len(p.Env) > 0 {
		env := os.Environ()
		for _, k := range slices.Sorted(maps.Keys(p.Env)) {
			env = append(env, k+"="+p.Env[k])
		}

		cmd.Env = env
	}

	cmd.Stdin = strings.NewReader(p.Stdin)
	cmd.Stdout = stdout
	cmd.Stderr = &out

	if outPath != "" {
		cmd.Stderr = &errOut
	}

	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = 2 * time.Second

	start := time.Now()
	err := cmd.Run()
	elapsed := time.Since(start).Round(time.Millisecond)

	if cmd.Process != nil &&
		(runCtx.Err() != nil || errors.Is(err, exec.ErrWaitDelay)) {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}

	var status string

	failed := true

	switch {
	case ctx.Err() != nil:
		status = "killed (run canceled)"
	case errors.Is(runCtx.Err(), context.DeadlineExceeded):
		status = fmt.Sprintf("killed after %s (timeout)", timeout)
	case err == nil:
		status, failed = "0", false
	case errors.Is(err, exec.ErrWaitDelay):
		code := -1
		if cmd.ProcessState != nil {
			code = cmd.ProcessState.ExitCode()
		}

		failed = code != 0
		status = fmt.Sprintf(
			"%d (exited, but background processes held the output open "+
				"and were killed)", code)
	default:
		if exitErr, ok := errors.AsType[*exec.ExitError](err); ok {
			status = strconv.Itoa(exitErr.ExitCode())
		} else {
			status = fmt.Sprintf("failed to start: %v", err)
		}
	}

	body := fmt.Sprintf("exit_code: %s\nduration: %s\n", status, elapsed)

	if outPath != "" {
		if fi, statErr := os.Stat(outPath); statErr == nil {
			written = fi.Size() - before
		}

		body += fmt.Sprintf("wrote %d bytes of stdout to %s\n", written, outPath)

		text := "(no stderr)"
		if errOut.Len() > 0 {
			text = strings.ToValidUTF8(errOut.String(), "�")
		}

		body += "--- stderr ---\n" + text

		if failed {
			return tool.NewTextErrorResponse(body), nil
		}

		return tool.NewTextResponse(body), nil
	}

	text := "(no output)"
	if out.Len() > 0 {
		text = strings.ToValidUTF8(out.String(), "�")
	}

	body += "--- stdout+stderr (combined) ---\n" + text
	if failed {
		return tool.NewTextErrorResponse(body), nil
	}

	return tool.NewTextResponse(body), nil
}

func newExecTool(cfg *Config, out io.Writer) tool.BaseTool {
	dir := cfg.Shell.Dir
	if dir == "" {
		dir, _ = os.Getwd()
	}

	return functiontool.New("exec", strings.TrimSpace(execToolDescription), (&execTool{
		shell:      cfg.Shell.Program,
		dir:        dir,
		timeout:    time.Duration(cfg.Shell.CommandTimeout),
		maxTimeout: time.Duration(cfg.Shell.MaxCommandTimeout),
		echo:       out,
	}).run)
}
