// SPDX-License-Identifier: Unlicense

package main

import (
	"bytes"
	"cmp"
	"context"
	"database/sql"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode"
	"uuid"

	"github.com/joakimcarlsson/ai/agent"
	"github.com/joakimcarlsson/ai/llm"
	"github.com/joakimcarlsson/ai/memory/sqlite"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/prompt"
	"github.com/joakimcarlsson/ai/schema"
	"github.com/joakimcarlsson/ai/session"
	"github.com/joakimcarlsson/ai/tokens"
	"github.com/joakimcarlsson/ai/tokens/truncate"
	"github.com/joakimcarlsson/ai/tool"
	"github.com/joakimcarlsson/ai/tool/functiontool"
	yaml "go.yaml.in/yaml/v3"
	_ "modernc.org/sqlite"
	_ "modernc.org/sqlite/vec"
)

const usage = `Usage
    echo '<goal>' | agent <session-uuid>
    agent <session-uuid>
    agent --init
    agent --help

A goal on stdin starts a session under that UUID. Omit the goal to resume a
session that already exists, which replays its original goal and carries on.
Piping a goal at a UUID that already holds a transcript is refused. Any UUID
will do for a new run, and the kernel hands you one at
/proc/sys/kernel/random/uuid.

--init writes the example config to $AGENT_CONFIG, or to the user config
directory when that is unset, and refuses if a file is already there.

Full trust, meaning no sandbox, no denylist and no confirmation prompts.

Configuration is read from that same file.
`

const execToolDescription = `Run one program and get back the exit code with the combined stdout and stderr.
The command goes through bash -c by default, so set program to python3, node, perl or anything else on PATH when the step is easier to write in that language than to quote through a shell.
Set program_flag alongside it when that interpreter does not take its program under -c, so -e for node, perl and ruby, where -c means check rather than run.
Every call is a fresh process, so nothing carries over. Use dir and env rather than cd and export, and in a shell chain dependent steps with &&.
To write a file, set command to "cat > /path" and put the contents in stdin, which avoids heredoc quoting.
Input you do not supply reads as EOF, but a command waiting for an interactive answer hangs until killed, so use non-interactive flags.
A non-zero exit is reported as an error, though grep, diff and test use non-zero normally, so judge by the output too.
Output is untruncated and stays in your context for the rest of the run, so pipe anything long through head, tail, wc -l or grep.
For output too large to read, set output_file. Stdout then goes to that file and you get back only a byte count, while stderr is still returned so a failure is still readable. Add append to keep what the file already holds, and mkdir to create its parent directory.`

const rememberToolDescription = `Search everything you have already done in this run and get back short descriptions of the matching turns.
Older turns are dropped from what you can still see once the conversation grows, but every one of them stays searchable here, so use this to recover work that has scrolled away.
Set query to describe what you are looking for in your own words, since the search goes by meaning rather than exact wording.
Each hit gives you an id, a relevance score from 0 to 1, when it happened in both clock time and age, the size of the original text, and a few lines saying what was attempted and what came of it.
Set id instead of query to get one entry back in full, including the complete command output as you first saw it.
Recent turns are still in front of you, so reach for this when you need something older than what you can read.`

const (
	functionType      = "function"
	maxBackoff        = 5 * time.Minute
	maxBackoffShift   = 4
	maxSearchLimit    = 25
	maxSummaryBytes   = 1000
	maxTokenizerCost  = 1 << 28
	rememberTries     = 3
	retryBackoff      = 30 * time.Second
	searchHeaderBytes = 160
	textType          = "text"
	typeKey           = "type"
)

//go:embed agent.yaml
var exampleConfig string

var errNoGoal = errors.New("no goal recorded in this session")

var envReference = regexp.MustCompile(
	`^(\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*\})$`)

type duration time.Duration

type EndpointConfig struct {
	Headers         map[string]string `yaml:"headers"`
	Params          map[string]any    `yaml:"params"`
	BaseURL         string            `yaml:"base_url"`
	APIKey          string            `yaml:"api_key"`
	Model           string            `yaml:"model"`
	ContextWindow   int64             `yaml:"context_window"`
	ContextFraction float64           `yaml:"context_fraction"`
	MaxTokens       int64             `yaml:"max_tokens"`
	RequestTimeout  duration          `yaml:"request_timeout"`
	RequestDelay    duration          `yaml:"request_delay"`
	MaxRetries      int               `yaml:"max_retries"`
}

type BudgetConfig struct {
	ExhaustedMessage string   `yaml:"exhausted_message"`
	MaxIterations    int      `yaml:"max_iterations"`
	Deadline         duration `yaml:"deadline"`
}

type ShellConfig struct {
	Program           string   `yaml:"program"`
	Dir               string   `yaml:"dir"`
	CommandTimeout    duration `yaml:"command_timeout"`
	MaxCommandTimeout duration `yaml:"max_command_timeout"`
}

type SessionConfig struct {
	Dir string `yaml:"dir"`
}

type RememberConfig struct {
	SummaryParams        map[string]any `yaml:"summary_params"`
	EmbeddingParams      map[string]any `yaml:"embedding_params"`
	EmbeddingQueryParams map[string]any `yaml:"embedding_query_params"`
	EmbeddingModel       string         `yaml:"embedding_model"`
	SummaryPrompt        string         `yaml:"summary_prompt"`
	SummaryModel         string         `yaml:"summary_model"`
	Dimensions           int            `yaml:"dimensions"`
	TopK                 int            `yaml:"top_k"`
	MaxFetchBytes        int            `yaml:"max_fetch_bytes"`
}

type Config struct {
	SystemPrompt string         `yaml:"system_prompt"`
	Session      SessionConfig  `yaml:"session"`
	Budget       BudgetConfig   `yaml:"budget"`
	Shell        ShellConfig    `yaml:"shell"`
	Remember     RememberConfig `yaml:"remember"`
	Endpoint     EndpointConfig `yaml:"endpoint"`
}

type pacer struct {
	begun bool
	delay time.Duration
	after time.Duration
	back  time.Duration
	waits int
	total time.Duration
}

type endpointAPI struct {
	client  *http.Client
	p       *pacer
	echo    io.Writer
	headers map[string]string
	base    string
	key     string
	retries int
	timeout time.Duration
}

type chatClient struct {
	api    *endpointAPI
	params map[string]any
	label  string
	model  llm.Model
	maxTok int64
}

type embedder struct {
	api     *endpointAPI
	passage map[string]any
	query   map[string]any
	model   string
	dims    int
}

type chatToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatMessage struct {
	Content    any            `json:"content,omitempty"`
	Role       string         `json:"role"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
}

type chatChoice struct {
	FinishReason string `json:"finish_reason"`
	Message      struct {
		Content          string         `json:"content"`
		Reasoning        string         `json:"reasoning"`
		ReasoningContent string         `json:"reasoning_content"`
		ToolCalls        []chatToolCall `json:"tool_calls"`
	} `json:"message"`
}

type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Usage   struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		PromptDetails    struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
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

type window struct {
	counter  *tokens.Counter
	trimmer  tokens.Strategy
	echo     io.Writer
	budget   int64
	sent     int64
	dropped  int
	turn     int
	maxTurns int
	opened   bool
}

type execTool struct {
	echo       io.Writer
	shell      string
	dir        string
	timeout    time.Duration
	maxTimeout time.Duration
}

type rememberParams struct {
	ID    *int64 `json:"id,omitempty" desc:"Return one entry in full instead of searching. Use an id from an earlier search result. Ignores query when set."`
	Limit *int   `json:"limit,omitempty" desc:"How many matches to return. Omit to use the configured default."`
	Query string `json:"query,omitempty" desc:"What to look for, described in your own words. Matched by meaning against a short description of every past turn, so a sentence works better than a keyword."`
}

type rememberIndex struct {
	db           *sql.DB
	embedder     *embedder
	llm          llm.LLM
	echo         io.Writer
	open         func()
	unsummarized map[int64]int
	sessionID    string
	prompt       string
	model        string
	topK         int
	maxFetch     int
	tokens       int64
}

type rememberPending struct {
	summary sql.Null[string]
	id      int64
	lo, hi  int64
}

type rememberSession struct {
	session.Session

	idx *rememberIndex
}

type rememberStore struct {
	inner session.Store
	idx   *rememberIndex
}

func (d *duration) UnmarshalYAML(n *yaml.Node) error {
	if n.Tag == "!!int" {
		var secs int64
		if err := n.Decode(&secs); err != nil {
			return fmt.Errorf("line %d: %w", n.Line, err)
		}

		*d = duration(time.Duration(secs) * time.Second)

		return nil
	}

	var s string
	if err := n.Decode(&s); err != nil {
		return fmt.Errorf("line %d: want a duration such as 90s, 2m or 1h30m, "+
			"or a plain number of seconds", n.Line)
	}

	v, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("line %d: %q is not a duration "+
			"(want e.g. 90s, 2m, 1h30m, or a plain number of seconds)", n.Line, s)
	}

	*d = duration(v)

	return nil
}

func (p *pacer) wait(ctx context.Context) {
	first := !p.begun
	p.begun = true

	server := p.after
	d := max(p.delay, server, p.back)
	p.after, p.back = 0, 0

	if first {
		return
	}

	if d <= 0 {
		return
	}

	start := time.Now()

	t := time.NewTimer(d)
	defer t.Stop()

	select {
	case <-ctx.Done():
	case <-t.C:
	}

	slept := time.Since(start)
	if rest := server - slept; rest > 0 {
		p.after = rest
	}

	p.waits++
	p.total += slept
}

func (a *endpointAPI) retryAfter(h http.Header) time.Duration {
	if ms := h.Get("Retry-After-Ms"); ms != "" {
		if n, err := strconv.Atoi(ms); err == nil && n > 0 {
			return time.Duration(n) * time.Millisecond
		}
	}

	v := h.Get("Retry-After")
	if v == "" {
		return 0
	}

	if n, err := strconv.Atoi(v); err == nil {
		return max(time.Duration(n)*time.Second, 0)
	}

	if t, err := http.ParseTime(v); err == nil {
		return max(time.Until(t), 0)
	}

	return 0
}

func (a *endpointAPI) attempt(
	ctx context.Context,
	path string,
	body []byte,
	out any,
) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimSuffix(a.base, "/")+path, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}

	req.Header.Set("Content-Type", "application/json")

	if a.key != "" {
		req.Header.Set("Authorization", "Bearer "+a.key)
	}

	for _, k := range slices.Sorted(maps.Keys(a.headers)) {
		req.Header.Set(k, a.headers[k])
	}

	resp, err := a.client.Do(req)
	if err != nil {
		return 0, err
	}

	defer func() { _ = resp.Body.Close() }()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<26))
	if err != nil {
		return resp.StatusCode, err
	}

	if d := a.retryAfter(resp.Header); d > 0 {
		a.p.after = max(a.p.after, min(d, maxBackoff))
	}

	if resp.StatusCode != http.StatusOK {
		return resp.StatusCode, fmt.Errorf("%s: %s: %s", path, resp.Status,
			strings.TrimSpace(string(raw)))
	}

	return resp.StatusCode, json.Unmarshal(raw, out)
}

func (a *endpointAPI) retryable(status int) bool {
	switch status {
	case 0, http.StatusRequestTimeout, http.StatusConflict,
		http.StatusTooManyRequests:
		return true
	default:
		return status >= http.StatusInternalServerError
	}
}

func oneBlock(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")

	kept := lines[:0]

	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			kept = append(kept, l)
		}
	}

	return strings.Join(kept, "\n")
}

func took(d time.Duration) time.Duration {
	if d >= time.Second {
		return d.Round(time.Second)
	}

	return d.Round(time.Millisecond)
}

func (a *endpointAPI) call(
	ctx context.Context,
	path, what string,
	in, out any,
) (err error) {
	body, err := json.Marshal(in)
	if err != nil {
		return err
	}

	start := time.Now()

	defer func() {
		state := "ok"
		if err != nil {
			state = "error"
		}

		_, _ = fmt.Fprintf(a.echo, "[api] %s -> %s (%s)\n",
			what, state, took(time.Since(start)))
	}()

	a.p.wait(ctx)

	if a.timeout > 0 {
		var cancel context.CancelFunc

		ctx, cancel = context.WithTimeout(ctx, a.timeout)
		defer cancel()
	}

	for attempt := 0; ; attempt++ {
		start = time.Now()

		status, err := a.attempt(ctx, path, body, out)
		if err == nil || attempt >= a.retries || ctx.Err() != nil ||
			!a.retryable(status) {
			return err
		}

		wait := max(a.p.delay, a.p.after,
			min(retryBackoff<<min(attempt, maxBackoffShift), maxBackoff))

		if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) <= wait {
			_, _ = fmt.Fprintf(a.echo,
				"[retry] %s skipped attempt %d of %d, %s wait does not fit "+
					"the %s left\n",
				what, attempt+1, a.retries, took(wait),
				took(time.Until(deadline)))

			return err
		}

		a.p.back = wait

		_, _ = fmt.Fprintf(a.echo, "[retry] %s attempt %d of %d in %s, %v\n",
			what, attempt+1, a.retries, took(wait), err)

		a.p.wait(ctx)
	}
}

func (c *chatClient) Model() llm.Model { return c.model }

func (c *chatClient) SupportsStructuredOutput() bool { return false }

func (c *chatClient) StreamResponse(
	context.Context, []message.Message, []tool.BaseTool,
) <-chan llm.Event {
	ch := make(chan llm.Event)
	close(ch)

	return ch
}

func (c *chatClient) StreamResponseWithStructuredOutput(
	context.Context, []message.Message, []tool.BaseTool,
	*schema.StructuredOutputInfo,
) <-chan llm.Event {
	ch := make(chan llm.Event)
	close(ch)

	return ch
}

func (c *chatClient) SendMessagesWithStructuredOutput(
	context.Context, []message.Message, []tool.BaseTool,
	*schema.StructuredOutputInfo,
) (*llm.Response, error) {
	return nil, errors.New("structured output is not supported")
}

func (c *chatClient) convert(msgs []message.Message) []chatMessage {
	var out []chatMessage

	for i := range msgs {
		m := &msgs[i]

		switch m.Role {
		case message.System:
			out = append(out, chatMessage{
				Role: "system", Content: m.Content().String(),
			})
		case message.User, message.Summary:
			out = append(out, chatMessage{Role: "user", Content: []any{
				map[string]any{typeKey: textType, textType: m.Content().String()},
			}})
		case message.Assistant:
			a := chatMessage{Role: "assistant"}
			if t := m.Content().String(); t != "" {
				a.Content = t
			}

			for _, call := range m.ToolCalls() {
				c := chatToolCall{ID: call.ID, Type: functionType}
				c.Function.Name = call.Name
				c.Function.Arguments = call.Input
				a.ToolCalls = append(a.ToolCalls, c)
			}

			out = append(out, a)
		case message.Tool:
			for _, r := range m.ToolResults() {
				out = append(out, chatMessage{
					Role: "tool", Content: r.Content, ToolCallID: r.ToolCallID,
				})
			}
		}
	}

	return out
}

func (c *chatClient) schemas(tools []tool.BaseTool) []map[string]any {
	out := make([]map[string]any, len(tools))

	for i, t := range tools {
		info := t.Info()
		params := map[string]any{
			typeKey: "object", "properties": info.Parameters,
		}

		if len(info.Required) > 0 {
			params["required"] = info.Required
		}

		out[i] = map[string]any{typeKey: functionType, functionType: map[string]any{
			"name":        info.Name,
			"description": info.Description,
			"parameters":  params,
		}}
	}

	return out
}

func (c *chatClient) finish(reason string, calls int) message.FinishReason {
	if calls > 0 {
		return message.FinishReasonToolUse
	}

	switch reason {
	case "stop":
		return message.FinishReasonEndTurn
	case "length":
		return message.FinishReasonMaxTokens
	case "tool_calls":
		return message.FinishReasonToolUse
	default:
		return message.FinishReasonUnknown
	}
}

func (c *chatClient) SendMessages(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
) (*llm.Response, error) {
	req := map[string]any{}
	if c.maxTok > 0 {
		req["max_completion_tokens"] = c.maxTok
	}

	maps.Copy(req, c.params)

	req["model"] = c.model.APIModel
	req["messages"] = c.convert(messages)
	req["stream"] = false

	if len(tools) > 0 {
		req["tools"] = c.schemas(tools)
	} else {
		delete(req, "tools")
		delete(req, "tool_choice")
		delete(req, "parallel_tool_calls")
		delete(req, "functions")
		delete(req, "function_call")
		delete(req, "tool_resources")
	}

	var out chatResponse
	if err := c.api.call(
		ctx, "/chat/completions", c.label, req, &out); err != nil {
		return nil, err
	}

	if len(out.Choices) == 0 {
		return nil, errors.New("no choices returned by the endpoint")
	}

	choice := out.Choices[0]

	var calls []message.ToolCall

	for _, t := range choice.Message.ToolCalls {
		calls = append(calls, message.ToolCall{
			ID: t.ID, Name: t.Function.Name, Input: t.Function.Arguments,
			Type: functionType, Finished: true,
		})
	}

	return &llm.Response{
		Content:      choice.Message.Content,
		Reasoning:    cmp.Or(choice.Message.Reasoning, choice.Message.ReasoningContent),
		ToolCalls:    calls,
		FinishReason: c.finish(choice.FinishReason, len(calls)),
		Usage: llm.TokenUsage{
			InputTokens: max(out.Usage.PromptTokens-
				out.Usage.PromptDetails.CachedTokens, 0),
			OutputTokens:    out.Usage.CompletionTokens,
			CacheReadTokens: out.Usage.PromptDetails.CachedTokens,
			ReasoningTokens: out.Usage.CompletionDetails.ReasoningTokens,
		},
	}, nil
}

func (e *embedder) embed(
	ctx context.Context,
	texts []string,
	extra map[string]any,
) ([][]float32, error) {
	req := map[string]any{}
	if e.dims > 0 {
		req["dimensions"] = e.dims
	}

	maps.Copy(req, extra)

	req["model"] = e.model
	req["input"] = texts

	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := e.api.call(ctx, "/embeddings", "embed", req, &out); err != nil {
		return nil, err
	}

	vecs := make([][]float32, len(out.Data))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("embeddings: index %d out of range", d.Index)
		}

		vecs[d.Index] = d.Embedding
	}

	return vecs, nil
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
		timeout = min(time.Duration(*p.TimeoutSeconds)*time.Second, t.maxTimeout)
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

func (w *window) tokenizerCost(s string) int64 {
	var total, run int64

	for _, r := range s {
		if unicode.IsSpace(r) {
			total += run * run
			run = 0

			continue
		}

		run++
	}

	return total + run*run
}

func (w *window) elideOversized(msgs []message.Message) []message.Message {
	out := slices.Clone(msgs)
	for i := range out {
		if out[i].Role != message.Tool {
			continue
		}

		var size, cost int64

		for _, r := range out[i].ToolResults() {
			size += int64(len(r.Content))
			cost += w.tokenizerCost(r.Content)
		}

		if size <= w.budget && cost <= maxTokenizerCost {
			continue
		}

		replaced := message.Message{
			Role:      out[i].Role,
			Model:     out[i].Model,
			CreatedAt: out[i].CreatedAt,
		}

		why := fmt.Sprintf("more than a %d-token budget can be sure of holding",
			w.budget)
		if size <= w.budget {
			why = "too dense to tokenize, being mostly one unbroken line"
		}

		for _, r := range out[i].ToolResults() {
			r.Content = fmt.Sprintf(
				"[output omitted: %d bytes, %s. Re-run piping through head, "+
					"tail or grep.]", size, why)
			r.IsError = true

			replaced.AddToolResult(r)
		}

		out[i] = replaced
	}

	return out
}

func (w *window) splitPinned(
	msgs []message.Message,
) (pinned, rest []message.Message) {
	i := 0
	for i < len(msgs) && msgs[i].Role == message.System {
		i++
	}

	if i < len(msgs) && msgs[i].Role == message.User {
		i++
	}

	return msgs[:i], msgs[i:]
}

func (w *window) openTurn() {
	w.turn++

	if w.turn <= w.maxTurns {
		_, _ = fmt.Fprintf(w.echo, "\n[turn %d/%d] %s\n",
			w.turn, w.maxTurns, time.Now().Format(time.RFC3339))
	}
}

func (w *window) first() {
	if w.turn == 0 {
		w.openTurn()

		w.opened = true
	}
}

func (w *window) preModelCall(
	ctx context.Context,
	mc agent.ModelCallContext,
) (agent.ModelCallResult, error) {
	if w.opened {
		w.opened = false
	} else {
		w.openTurn()
	}

	pinned, rest := w.splitPinned(w.elideOversized(mc.Messages))

	remaining := w.budget
	if c, err := w.counter.CountTokens(ctx, tokens.CountOptions{
		Messages: pinned,
		Tools:    mc.Tools,
	}); err == nil {
		remaining = max(w.budget-c.TotalTokens, 512)
	}

	res, err := w.trimmer.Fit(ctx, tokens.StrategyInput{
		Messages:  rest,
		Tools:     mc.Tools,
		Counter:   w.counter,
		MaxTokens: remaining,
	})
	if err != nil {
		return agent.ModelCallResult{}, err
	}

	out := slices.Concat(pinned, res.Messages)

	w.dropped += len(rest) - len(res.Messages)
	if c, err := w.counter.CountTokens(ctx, tokens.CountOptions{
		Messages: out,
		Tools:    mc.Tools,
	}); err == nil {
		w.sent = c.TotalTokens
	}

	return agent.ModelCallResult{
		Action:   agent.HookModify,
		Messages: out,
		Tools:    mc.Tools,
	}, nil
}

func (x *rememberIndex) text(msgs []message.Message) string {
	var b strings.Builder

	for i := range msgs {
		m := &msgs[i]
		for _, part := range m.Parts {
			switch c := part.(type) {
			case message.TextContent:
				if t := strings.TrimSpace(c.Text); t != "" {
					b.WriteString(string(m.Role) + ": " + t + "\n")
				}
			case message.ReasoningContent:
				if t := strings.TrimSpace(c.Text); t != "" {
					b.WriteString("reasoning: " + t + "\n")
				}
			case message.ToolCall:
				b.WriteString("called " + c.Name + " with " + c.Input + "\n")
			case message.ToolResult:
				verb := "result of "
				if c.IsError {
					verb = "failed result of "
				}

				b.WriteString(verb + c.Name + "\n" + c.Content + "\n")

				if md := strings.TrimSpace(c.Metadata); md != "" {
					b.WriteString("metadata: " + md + "\n")
				}
			case message.ImageURLContent:
				b.WriteString("image: " + c.URL + " " + c.Detail + "\n")
			case message.BinaryContent:
				fmt.Fprintf(&b, "binary: %s %s %d bytes\n",
					c.Path, c.MIMEType, len(c.Data))
			default:
				fmt.Fprintf(&b, "part: %T\n", c)
			}
		}
	}

	return strings.TrimSpace(b.String())
}

func (x *rememberIndex) maxID(ctx context.Context) int64 {
	var id sql.Null[int64]
	if err := x.db.QueryRowContext(ctx,
		`SELECT MAX(id) FROM messages WHERE session_id = ?`,
		x.sessionID).Scan(&id); err != nil {
		return 0
	}

	return id.V
}

func (x *rememberIndex) body(
	ctx context.Context,
	lo, hi int64,
) (string, error) {
	rows, err := x.db.QueryContext(ctx, `SELECT parts FROM messages
		WHERE session_id = ? AND id BETWEEN ? AND ? ORDER BY id`,
		x.sessionID, lo, hi)
	if err != nil {
		return "", err
	}

	defer func() { _ = rows.Close() }()

	var msgs []message.Message

	for rows.Next() {
		var blob string
		if err := rows.Scan(&blob); err != nil {
			return "", err
		}

		var m message.Message
		if err := json.Unmarshal([]byte(blob), &m); err != nil {
			continue
		}

		msgs = append(msgs, m)
	}

	return x.text(msgs), rows.Err()
}

func (x *rememberIndex) brief(s string) string {
	if len(s) <= maxSummaryBytes {
		return s
	}

	return strings.ToValidUTF8(s[:maxSummaryBytes], "")
}

func (x *rememberIndex) summarize(
	ctx context.Context,
	body string,
) (string, error) {
	const head, tail = 8000, 8000

	if len(body) > head+tail {
		body = strings.ToValidUTF8(body[:head], "") + "\n[...]\n" +
			strings.ToValidUTF8(body[len(body)-tail:], "")
	}

	resp, err := x.llm.SendMessages(ctx, []message.Message{
		message.NewSystemMessage(x.prompt),
		message.NewUserMessage(body),
	}, nil)
	if err != nil {
		return "", err
	}

	x.tokens += resp.Usage.InputTokens + resp.Usage.OutputTokens

	s := strings.TrimSpace(resp.Content)
	if s == "" {
		return "", errors.New("model returned an empty summary")
	}

	return x.brief(s), nil
}

func (x *rememberIndex) embed(
	ctx context.Context,
	texts []string,
	extra map[string]any,
) ([][]float32, error) {
	vecs, err := x.embedder.embed(ctx, texts, extra)
	if err != nil {
		return nil, err
	}

	if len(vecs) != len(texts) {
		return nil, fmt.Errorf("got %d vectors for %d inputs",
			len(vecs), len(texts))
	}

	return vecs, nil
}

func (x *rememberIndex) embedPassage(
	ctx context.Context,
	texts []string,
) ([][]float32, error) {
	return x.embed(ctx, texts, x.embedder.passage)
}

func (x *rememberIndex) embedQuery(
	ctx context.Context,
	texts []string,
) ([][]float32, error) {
	return x.embed(ctx, texts, x.embedder.query)
}

func (x *rememberIndex) counts(ctx context.Context) (total, ready int) {
	_ = x.db.QueryRowContext(ctx,
		`SELECT count(*), coalesce(sum(vector IS NOT NULL AND model = ?), 0)
		FROM remember WHERE session_id = ?`,
		x.model, x.sessionID).Scan(&total, &ready)

	return total, ready
}

func (x *rememberIndex) pending(
	ctx context.Context,
) ([]rememberPending, error) {
	rows, err := x.db.QueryContext(ctx,
		`SELECT id, lo_id, hi_id, summary FROM remember
		WHERE session_id = ? AND vector IS NULL ORDER BY id`, x.sessionID)
	if err != nil {
		return nil, err
	}

	defer func() { _ = rows.Close() }()

	var todo []rememberPending

	for rows.Next() {
		var p rememberPending
		if err := rows.Scan(&p.id, &p.lo, &p.hi, &p.summary); err != nil {
			return nil, err
		}

		todo = append(todo, p)
	}

	return todo, rows.Err()
}

func (x *rememberIndex) fill(ctx context.Context) {
	save := context.WithoutCancel(ctx)

	todo, listErr := x.pending(ctx)
	if listErr != nil {
		_, _ = fmt.Fprintf(x.echo, "[remember] %v\n", listErr)

		return
	}

	var (
		ids       []int64
		summaries []string
	)

	for _, p := range todo {
		if p.summary.Valid {
			ids = append(ids, p.id)
			summaries = append(summaries, p.summary.V)

			continue
		}

		if x.unsummarized[p.id] >= rememberTries {
			continue
		}

		body, err := x.body(ctx, p.lo, p.hi)
		if err != nil || body == "" {
			continue
		}

		s, err := x.summarize(ctx, body)
		if err != nil {
			x.unsummarized[p.id]++

			_, _ = fmt.Fprintf(x.echo, "[remember] summary failed: %v\n", err)

			continue
		}

		if _, err := x.db.ExecContext(save,
			`UPDATE remember SET summary = ? WHERE id = ?`, s, p.id); err != nil {
			_, _ = fmt.Fprintf(x.echo, "[remember] %v\n", err)
			continue
		}

		ids = append(ids, p.id)
		summaries = append(summaries, s)
	}

	if len(ids) == 0 {
		return
	}

	vecs, err := x.embedPassage(ctx, summaries)
	if err != nil {
		_, _ = fmt.Fprintf(x.echo, "[remember] embedding failed: %v\n", err)
		return
	}

	for i, id := range ids {
		blob, err := binary.Append(nil, binary.LittleEndian, vecs[i])
		if err != nil {
			_, _ = fmt.Fprintf(x.echo, "[remember] %v\n", err)
			continue
		}

		if _, err := x.db.ExecContext(save, `UPDATE remember
			SET summary = ?, vector = ?, dims = ?, model = ? WHERE id = ?`,
			summaries[i], blob, len(vecs[i]), x.model, id); err != nil {
			_, _ = fmt.Fprintf(x.echo, "[remember] %v\n", err)
		}
	}
}

func (x *rememberIndex) add(
	ctx context.Context,
	msgs []message.Message,
	lo, hi int64,
) {
	calls, mine := 0, 0

	for i := range msgs {
		for _, c := range msgs[i].ToolCalls() {
			calls++

			if c.Name == "remember" {
				mine++
			}
		}
	}

	if calls > 0 && calls == mine {
		return
	}

	body := x.text(msgs)
	if body == "" {
		return
	}

	if _, err := x.db.ExecContext(ctx, `INSERT INTO remember
		(session_id, lo_id, hi_id, bytes, created_at, model, dims)
		VALUES (?, ?, ?, ?, ?, ?, 0)`,
		x.sessionID, lo, hi, len(body), time.Now().UnixNano(),
		x.model); err != nil {
		_, _ = fmt.Fprintf(x.echo, "[remember] %v\n", err)

		return
	}

	x.fill(ctx)
}

func (x *rememberIndex) search(
	ctx context.Context,
	query string,
	limit int,
) (string, error) {
	vecs, err := x.embedQuery(ctx, []string{query})
	if err != nil {
		return "", err
	}

	q, err := binary.Append(nil, binary.LittleEndian, vecs[0])
	if err != nil {
		return "", err
	}

	rows, err := x.db.QueryContext(ctx,
		`SELECT id, created_at, bytes, summary,
			coalesce(1 - vec_distance_cosine(vector, ?), 0) AS score
		FROM remember
		WHERE session_id = ? AND model = ? AND dims = ? AND vector IS NOT NULL
		ORDER BY score DESC, created_at DESC
		LIMIT ?`,
		q, x.sessionID, x.model, len(vecs[0]), limit)
	if err != nil {
		return "", err
	}

	defer func() { _ = rows.Close() }()

	type hit struct {
		summary string
		score   float64
		id      int64
		at      int64
		size    int64
	}

	hits := make([]hit, 0, limit)

	for rows.Next() {
		var h hit
		if err := rows.Scan(
			&h.id, &h.at, &h.size, &h.summary, &h.score); err != nil {
			return "", err
		}

		hits = append(hits, h)
	}

	if err := rows.Err(); err != nil {
		return "", err
	}

	if len(hits) == 0 {
		total, ready := x.counts(ctx)

		switch {
		case total == 0:
			return "Nothing has been indexed for this session yet.", nil
		case ready == 0:
			return "Nothing has been embedded with the current model yet, " +
				"so there is nothing to match against.", nil
		default:
			return "No embedded entry matches the vector size in use now.", nil
		}
	}

	now := time.Now()
	room := x.maxFetch - searchHeaderBytes

	var (
		entries []string
		used    int
	)

	for _, h := range hits {
		at := time.Unix(0, h.at)
		e := fmt.Sprintf("\n[%d] %.2f | %s | %s ago | %d bytes\n%s\n",
			h.id, h.score, at.Format(time.DateTime),
			now.Sub(at).Round(time.Second), h.size, x.brief(h.summary))

		if len(entries) > 0 && used+len(e) > room {
			break
		}

		entries = append(entries, e)
		used += len(e)
	}

	var out strings.Builder
	if len(entries) < len(hits) {
		fmt.Fprintf(&out, "Showing %d of %d matches, most relevant first. "+
			"Narrow the query or lower limit to see the rest. "+
			"Pass id to get one back in full.\n", len(entries), len(hits))
	} else {
		fmt.Fprintf(&out, "%d matches, most relevant first. "+
			"Pass id to get one back in full.\n", len(hits))
	}

	for _, e := range entries {
		out.WriteString(e)
	}

	return strings.TrimSpace(out.String()), nil
}

func (x *rememberIndex) affordable(fixed, s string) string {
	var total, run int64

	for _, r := range fixed {
		if unicode.IsSpace(r) {
			total += run * run
			run = 0

			continue
		}

		run++
	}

	total += run * run
	run = 0

	for i, r := range s {
		if unicode.IsSpace(r) {
			total += run * run
			run = 0

			continue
		}

		run++

		if total+run*run >= maxTokenizerCost {
			return s[:i]
		}
	}

	return s
}

func (x *rememberIndex) fetch(
	ctx context.Context,
	id int64,
) (string, error) {
	var (
		lo, hi, at int64
		summary    sql.Null[string]
	)

	if err := x.db.QueryRowContext(ctx,
		`SELECT lo_id, hi_id, created_at, summary FROM remember
		WHERE id = ? AND session_id = ?`, id, x.sessionID).
		Scan(&lo, &hi, &at, &summary); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", fmt.Errorf("no entry with id %d in this session", id)
		}

		return "", err
	}

	body, err := x.body(ctx, lo, hi)
	if err != nil {
		return "", err
	}

	when := time.Unix(0, at)
	out := fmt.Sprintf("[%d] %s | %s ago | %d bytes\n%s\n\n",
		id, when.Format(time.DateTime), time.Since(when).Round(time.Second),
		len(body), x.brief(summary.V))

	if len(out) >= x.maxFetch {
		return fmt.Sprintf("Entry %d is %d bytes and its description alone "+
			"exceeds max_fetch_bytes of %d, so none of it can be returned. "+
			"Raise max_fetch_bytes or widen the context window.",
			id, len(body), x.maxFetch), nil
	}

	room := x.maxFetch - len(out)

	cut := body
	if len(cut) > room {
		cut = strings.ToValidUTF8(cut[:room], "")
	}

	cut = x.affordable(out+fmt.Sprintf(
		"\n[truncated at %d of %d bytes]", len(body), len(body)), cut)

	if len(cut) == len(body) {
		return out + body, nil
	}

	trailer := fmt.Sprintf("\n[truncated at %d of %d bytes]", len(cut), len(body))

	if over := len(cut) + len(trailer) - room; over > 0 {
		cut = strings.ToValidUTF8(cut[:max(len(cut)-over, 0)], "")
		trailer = fmt.Sprintf(
			"\n[truncated at %d of %d bytes]", len(cut), len(body))
	}

	return out + cut + trailer, nil
}

func (x *rememberIndex) remember(
	ctx context.Context,
	p rememberParams,
) (tool.Response, error) {
	if p.ID != nil {
		out, err := x.fetch(ctx, *p.ID)
		if err != nil {
			return tool.NewTextErrorResponse(
				fmt.Sprintf("remember: %v", err)), nil
		}

		return tool.NewTextResponse(out), nil
	}

	query := strings.TrimSpace(p.Query)
	if query == "" {
		return tool.NewTextErrorResponse(
			`remember: set "query" to search, or "id" to fetch one entry.`), nil
	}

	limit := x.topK
	if p.Limit != nil && *p.Limit > 0 {
		limit = min(*p.Limit, maxSearchLimit)
	}

	out, err := x.search(ctx, query, limit)
	if err != nil {
		return tool.NewTextErrorResponse(
			fmt.Sprintf("remember: %v", err)), nil
	}

	return tool.NewTextResponse(out), nil
}

func (s *rememberSession) AddMessages(
	ctx context.Context,
	msgs []message.Message,
) error {
	if s.idx.open != nil {
		s.idx.open()
	}

	lo := s.idx.maxID(ctx) + 1

	if err := s.Session.AddMessages(ctx, msgs); err != nil {
		return err
	}

	if hi := s.idx.maxID(ctx); hi >= lo {
		s.idx.add(ctx, msgs, lo, hi)
	}

	return nil
}

func (s *rememberStore) Exists(
	ctx context.Context,
	id string,
) (bool, error) {
	return s.inner.Exists(ctx, id)
}

func (s *rememberStore) Create(
	ctx context.Context,
	id string,
) (session.Session, error) {
	inner, err := s.inner.Create(ctx, id)
	if err != nil {
		return nil, err
	}

	return &rememberSession{Session: inner, idx: s.idx}, nil
}

func (s *rememberStore) Load(
	ctx context.Context,
	id string,
) (session.Session, error) {
	inner, err := s.inner.Load(ctx, id)
	if err != nil {
		return nil, err
	}

	return &rememberSession{Session: inner, idx: s.idx}, nil
}

func (s *rememberStore) Delete(ctx context.Context, id string) error {
	if _, err := s.idx.db.ExecContext(ctx,
		`DELETE FROM remember WHERE session_id = ?`, id); err != nil {
		_, _ = fmt.Fprintf(s.idx.echo, "[remember] %v\n", err)
	}

	return s.inner.Delete(ctx, id)
}

func configPath() (string, error) {
	if path := os.Getenv("AGENT_CONFIG"); path != "" {
		return path, nil
	}

	base, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("no $AGENT_CONFIG and no user config directory "+
			"to fall back on: %w", err)
	}

	return filepath.Join(base, "agent", "config.yaml"), nil
}

func LoadConfig() (Config, string, error) {
	defaults := Config{
		Endpoint: EndpointConfig{
			BaseURL:         "https://api.openai.com/v1",
			ContextWindow:   128000,
			ContextFraction: 0.75,
			MaxRetries:      3,
		},
		Budget: BudgetConfig{
			MaxIterations: 200,
		},
		Session: SessionConfig{Dir: ".agent"},
		Shell: ShellConfig{
			Program:           "/bin/bash",
			CommandTimeout:    duration(2 * time.Minute),
			MaxCommandTimeout: duration(15 * time.Minute),
		},
		Remember: RememberConfig{TopK: 5, MaxFetchBytes: 65536},
	}
	cfg := defaults

	path, err := configPath()
	if err != nil {
		return cfg, "", err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, path, fmt.Errorf("reading config: %w", err)
	}

	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)

	if err := dec.Decode(&cfg); err != nil {
		return cfg, path, fmt.Errorf("%s: %w", path, err)
	}

	cfg.Endpoint.BaseURL = cmp.Or(cfg.Endpoint.BaseURL, defaults.Endpoint.BaseURL)

	cfg.Endpoint.MaxTokens = max(cfg.Endpoint.MaxTokens, 0)

	cfg.Endpoint.RequestTimeout = max(cfg.Endpoint.RequestTimeout, 0)
	cfg.Endpoint.RequestDelay = max(cfg.Endpoint.RequestDelay, 0)
	cfg.Endpoint.MaxRetries = max(cfg.Endpoint.MaxRetries, 0)

	if cfg.Endpoint.ContextWindow < 4096 {
		cfg.Endpoint.ContextWindow = defaults.Endpoint.ContextWindow
	}

	if cfg.Endpoint.ContextFraction <= 0 || cfg.Endpoint.ContextFraction > 1 {
		cfg.Endpoint.ContextFraction = defaults.Endpoint.ContextFraction
	}

	cfg.Budget.Deadline = max(cfg.Budget.Deadline, 0)

	if cfg.Budget.MaxIterations < 1 {
		cfg.Budget.MaxIterations = defaults.Budget.MaxIterations
	}

	cfg.Session.Dir = cmp.Or(cfg.Session.Dir, defaults.Session.Dir)

	cfg.Shell.Program = cmp.Or(cfg.Shell.Program, defaults.Shell.Program)
	if cfg.Shell.CommandTimeout <= 0 {
		cfg.Shell.CommandTimeout = defaults.Shell.CommandTimeout
	}

	if cfg.Shell.MaxCommandTimeout <= 0 {
		cfg.Shell.MaxCommandTimeout = defaults.Shell.MaxCommandTimeout
	}

	cfg.Shell.MaxCommandTimeout = max(cfg.Shell.MaxCommandTimeout, cfg.Shell.CommandTimeout)

	if raw := cfg.Endpoint.APIKey; envReference.MatchString(raw) {
		if cfg.Endpoint.APIKey = os.ExpandEnv(raw); cfg.Endpoint.APIKey == "" {
			return cfg, path, fmt.Errorf(
				"%s: endpoint.api_key is %q but that expands to nothing, so "+
					"give it a value, set the key literally, or use \"\" for "+
					"an endpoint that wants no auth", path, raw)
		}
	}

	cfg.Remember.Dimensions = max(cfg.Remember.Dimensions, 0)
	if cfg.Remember.TopK < 1 {
		cfg.Remember.TopK = defaults.Remember.TopK
	}

	cfg.Remember.TopK = min(cfg.Remember.TopK, maxSearchLimit)

	if cfg.Remember.MaxFetchBytes < 1 {
		cfg.Remember.MaxFetchBytes = defaults.Remember.MaxFetchBytes
	}

	cfg.Remember.MaxFetchBytes = min(cfg.Remember.MaxFetchBytes,
		int(float64(cfg.Endpoint.ContextWindow)*cfg.Endpoint.ContextFraction))

	if u, err := url.Parse(cfg.Endpoint.BaseURL); err != nil ||
		u.Scheme == "" || u.Host == "" {
		return cfg, path, fmt.Errorf(
			"%s: endpoint.base_url %q is not an absolute URL "+
				"(want e.g. http://localhost:11434/v1)", path, cfg.Endpoint.BaseURL)
	}

	if cfg.Endpoint.Model == "" {
		return cfg, path, fmt.Errorf("%s: endpoint.model is required", path)
	}

	if cfg.Budget.ExhaustedMessage == "" {
		return cfg, path, fmt.Errorf("%s: budget.exhausted_message is required", path)
	}

	if _, err := exec.LookPath(cfg.Shell.Program); err != nil {
		return cfg, path, fmt.Errorf("%s: shell.program %q: %w",
			path, cfg.Shell.Program, err)
	}

	if cfg.Shell.Dir != "" {
		if fi, err := os.Stat(cfg.Shell.Dir); err != nil || !fi.IsDir() {
			return cfg, path, fmt.Errorf("%s: shell.dir %q is not a directory",
				path, cfg.Shell.Dir)
		}
	}

	if cfg.Remember.EmbeddingModel != "" && cfg.Remember.SummaryPrompt == "" {
		return cfg, path, fmt.Errorf("%s: remember.summary_prompt is "+
			"required when remember.embedding_model is set", path)
	}

	return cfg, path, nil
}

func printHelp(out io.Writer) {
	_, _ = fmt.Fprint(out, usage)

	path, err := configPath()
	if err != nil {
		_, _ = fmt.Fprintf(out,
			"\nThe config location could not be resolved, %v\n", err)

		return
	}

	state := "does not exist yet, so run agent --init"
	if _, err := os.Stat(path); err == nil {
		state = "already exists"
	}

	_, _ = fmt.Fprintf(out, "\nConfig file\n    %s (%s)\n", path, state)
}

func initConfig(out io.Writer) int {
	path, err := configPath()
	if err != nil {
		_, _ = fmt.Fprintf(out, "agent: %v\n", err)

		return 1
	}

	if _, statErr := os.Stat(path); statErr == nil {
		_, _ = fmt.Fprintf(out, "agent: config already exists at %s\n\n"+
			"    edit it, or move it aside and rerun\n", path)

		return 1
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		_, _ = fmt.Fprintf(out, "agent: %v\n", err)

		return 1
	}

	if err := os.WriteFile(path, []byte(exampleConfig), 0o600); err != nil {
		_, _ = fmt.Fprintf(out, "agent: %v\n", err)

		return 1
	}

	_, _ = fmt.Fprintf(out, "wrote %s\n", path)

	return 0
}

func sessionIDFromArgs(args []string) (string, error) {
	if len(args) != 1 {
		return "", errors.New("expected exactly one argument, the session UUID")
	}

	u, err := uuid.Parse(args[0])
	if err != nil {
		return "", fmt.Errorf("%q is not a session UUID: %w", args[0], err)
	}

	return u.String(), nil
}

func readGoal() (string, bool, error) {
	stat, err := os.Stdin.Stat()
	if err != nil {
		return "", false, fmt.Errorf("inspecting stdin: %w", err)
	}

	if stat.Mode()&os.ModeCharDevice != 0 {
		return "", false, nil
	}

	b, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", false, fmt.Errorf("reading goal from stdin: %w", err)
	}

	goal := strings.TrimSpace(string(b))

	return goal, goal != "", nil
}

func sessionGoal(
	ctx context.Context,
	store session.Store,
	id string,
) (string, error) {
	s, err := store.Load(ctx, id)
	if err != nil {
		return "", err
	}

	msgs, err := s.GetMessages(ctx, nil)
	if err != nil {
		return "", err
	}

	for i := range msgs {
		if msgs[i].Role != message.User {
			continue
		}

		for _, part := range msgs[i].Parts {
			t, ok := part.(message.TextContent)
			if !ok {
				continue
			}

			if goal := strings.TrimSpace(t.Text); goal != "" {
				return goal, nil
			}
		}
	}

	return "", errNoGoal
}

func endpointHost(base string) string {
	if u, err := url.Parse(base); err == nil && u.Host != "" {
		return u.Host
	}

	return "custom"
}

func newClient(
	cfg *Config,
	api *endpointAPI,
	label, name string,
	params map[string]any,
) (llm.LLM, llm.Model) {
	model := llm.NewCustomModel(
		llm.WithModelID(name),
		llm.WithAPIModel(name),
		llm.WithName(name),
		llm.WithProvider(endpointHost(cfg.Endpoint.BaseURL)),
		llm.WithContextWindow(cfg.Endpoint.ContextWindow),
		llm.WithDefaultMaxTokens(cfg.Endpoint.MaxTokens),
	)

	return &chatClient{
		api: api, model: model, params: params, label: label,
		maxTok: cfg.Endpoint.MaxTokens,
	}, model
}

func logTool(
	out io.Writer,
) func(context.Context, agent.PostToolUseContext) (agent.PostToolUseResult, error) {
	return func(
		_ context.Context,
		tc agent.PostToolUseContext,
	) (agent.PostToolUseResult, error) {
		status := "ok"
		if tc.IsError {
			status = "error"
		}

		_, _ = fmt.Fprintf(out, "[tool] %s -> %s (%d bytes, %s)\n",
			tc.ToolName, status, len(tc.Output), took(tc.Duration))

		return agent.PostToolUseResult{}, nil
	}
}

func logReasoning(
	out io.Writer,
) func(context.Context, agent.ModelResponseContext) (agent.ModelResponseResult, error) {
	return func(
		_ context.Context,
		mc agent.ModelResponseContext,
	) (agent.ModelResponseResult, error) {
		if mc.Response == nil {
			return agent.ModelResponseResult{}, nil
		}

		if r := oneBlock(mc.Response.Reasoning); r != "" {
			_, _ = fmt.Fprintf(out, "[reasoning]\n%s\n", r)
		}

		return agent.ModelResponseResult{}, nil
	}
}

func exhaustedData(done, limit, pending int) map[string]any {
	return map[string]any{
		"TotalIterations":  done,
		"MaxIterations":    limit,
		"PendingToolCalls": pending,
	}
}

func budgetProvider(
	cfg *Config,
	exhausted *prompt.Template,
	out io.Writer,
) agent.ContinuationProvider {
	return func(
		_ context.Context,
		req agent.ContinuationRequest,
	) (agent.ContinuationResponse, error) {
		done := req.TotalIterations
		if done < cfg.Budget.MaxIterations {
			return agent.ContinuationResponse{
				Decision: agent.ContinuationApprove,
			}, nil
		}

		_, _ = fmt.Fprintf(out, "[budget] exhausted at %d/%d, wrapping up\n",
			done, cfg.Budget.MaxIterations)

		msg, err := exhausted.Process(exhaustedData(
			done, cfg.Budget.MaxIterations, len(req.ToolCalls)))
		if err != nil {
			return agent.ContinuationResponse{}, err
		}

		return agent.ContinuationResponse{
			Decision: agent.ContinuationDecline,
			Message:  msg,
		}, nil
	}
}

func newRememberIndex(
	ctx context.Context,
	cfg *Config,
	db *sql.DB,
	client llm.LLM,
	api *endpointAPI,
	sessionID string,
	out io.Writer,
) (*rememberIndex, error) {
	for _, stmt := range []string{
		`CREATE TABLE IF NOT EXISTS remember (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT    NOT NULL,
			lo_id      INTEGER NOT NULL,
			hi_id      INTEGER NOT NULL,
			bytes      INTEGER NOT NULL,
			created_at INTEGER NOT NULL,
			model      TEXT    NOT NULL,
			dims       INTEGER NOT NULL,
			summary    TEXT,
			vector     BLOB
		)`,
		`CREATE INDEX IF NOT EXISTS idx_remember_session
			ON remember(session_id, id)`,
	} {
		if _, err := db.ExecContext(ctx, stmt); err != nil {
			return nil, err
		}
	}

	query := map[string]any{}
	maps.Copy(query, cfg.Remember.EmbeddingParams)
	maps.Copy(query, cfg.Remember.EmbeddingQueryParams)

	return &rememberIndex{
		db: db,
		embedder: &embedder{
			api:     api,
			model:   cfg.Remember.EmbeddingModel,
			passage: cfg.Remember.EmbeddingParams,
			query:   query,
			dims:    cfg.Remember.Dimensions,
		},
		llm:          client,
		echo:         out,
		unsummarized: map[int64]int{},
		sessionID:    sessionID,
		prompt:       cfg.Remember.SummaryPrompt,
		model:        cfg.Remember.EmbeddingModel,
		topK:         cfg.Remember.TopK,
		maxFetch:     cfg.Remember.MaxFetchBytes,
	}, nil
}

func run() int {
	args := os.Args[1:]
	if len(args) == 1 {
		switch {
		case slices.Contains([]string{"-h", "--help", "help"}, args[0]):
			printHelp(os.Stderr)

			return 0
		case args[0] == "--init":
			return initConfig(os.Stderr)
		}
	}

	sessionID, err := sessionIDFromArgs(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: %v\n", err)
		fmt.Fprint(os.Stderr, usage)

		return 1
	}

	cfg, path, err := LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: %v\n", err)
		return 1
	}

	goal, hasGoal, err := readGoal()
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: %v\n", err)
		return 1
	}

	if _, statErr := os.Stat(
		filepath.Join(cfg.Session.Dir, sessionID+".db")); statErr != nil && !hasGoal {
		fmt.Fprintf(os.Stderr, "agent: no session %s in %s and no goal on stdin\n\n"+
			"    start it\n        echo '<goal>' | agent %s\n",
			sessionID, cfg.Session.Dir, sessionID)

		return 1
	}

	workDir := cfg.Shell.Dir
	if workDir == "" {
		workDir, _ = os.Getwd()
	}

	ctx, stop := signal.NotifyContext(
		context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	context.AfterFunc(ctx, stop)

	if d := time.Duration(cfg.Budget.Deadline); d > 0 {
		var cancel context.CancelFunc

		ctx, cancel = context.WithTimeout(ctx, d)
		defer cancel()
	}

	pace := &pacer{delay: time.Duration(cfg.Endpoint.RequestDelay)}
	api := &endpointAPI{
		client:  &http.Client{},
		p:       pace,
		echo:    os.Stderr,
		base:    cfg.Endpoint.BaseURL,
		key:     cfg.Endpoint.APIKey,
		headers: cfg.Endpoint.Headers,
		retries: cfg.Endpoint.MaxRetries,
		timeout: time.Duration(cfg.Endpoint.RequestTimeout),
	}

	client, model := newClient(
		&cfg, api, "chat", cfg.Endpoint.Model, cfg.Endpoint.Params)

	exhausted, err := prompt.New(cfg.Budget.ExhaustedMessage,
		prompt.WithName("budget.exhausted_message"), prompt.WithStrictMode())
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: budget.exhausted_message: %v\n", err)
		return 1
	}

	if _, err = exhausted.Process(exhaustedData(0, 0, 0)); err != nil {
		fmt.Fprintf(os.Stderr, "agent: budget.exhausted_message: %v\n", err)
		return 1
	}

	counter, err := tokens.NewCounter()
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: token counter: %v\n", err)
		return 1
	}

	w := &window{
		counter: counter,
		trimmer: truncate.Strategy(truncate.PreservePairs(), truncate.MinMessages(2)),
		echo:    os.Stderr,
		budget: int64(float64(cfg.Endpoint.ContextWindow) *
			cfg.Endpoint.ContextFraction),
		maxTurns: cfg.Budget.MaxIterations,
	}

	tools := []tool.BaseTool{
		functiontool.New("exec", execToolDescription, (&execTool{
			shell:      cfg.Shell.Program,
			dir:        workDir,
			timeout:    time.Duration(cfg.Shell.CommandTimeout),
			maxTimeout: time.Duration(cfg.Shell.MaxCommandTimeout),
			echo:       os.Stderr,
		}).run),
	}

	opts := []agent.Option{
		agent.WithSequentialToolExecution(),
		agent.WithHooks(agent.Hooks{
			PreModelCall:  w.preModelCall,
			PostModelCall: logReasoning(os.Stderr),
			PostToolUse:   logTool(os.Stderr),
		}),
		agent.WithMaxIterations(1),
		agent.WithContinuationProvider(
			budgetProvider(&cfg, exhausted, os.Stderr)),
	}

	if cfg.SystemPrompt != "" {
		opts = append(opts, agent.WithInstructionProvider(
			func(context.Context, map[string]any) (string, error) {
				return cfg.SystemPrompt, nil
			}))
	}

	if err = os.MkdirAll(cfg.Session.Dir, 0o700); err != nil {
		fmt.Fprintf(os.Stderr, "agent: %v\n", err)
		return 1
	}

	dbPath := filepath.Join(cfg.Session.Dir, sessionID+".db")

	db, err := sql.Open("sqlite", "file:"+dbPath+
		"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)")
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: opening %s: %v\n", dbPath, err)
		return 1
	}
	defer func() { _ = db.Close() }()

	store, err := sqlite.SessionStore(ctx, db)
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: %s: %v\n", dbPath, err)
		return 1
	}

	started, err := store.Exists(ctx, sessionID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: %s: %v\n", dbPath, err)
		return 1
	}

	var recorded string

	if started {
		recorded, err = sessionGoal(ctx, store, sessionID)

		switch {
		case errors.Is(err, errNoGoal):
			started = false
		case err != nil:
			fmt.Fprintf(os.Stderr, "agent: %s: %v\n", dbPath, err)
			return 1
		}
	}

	switch {
	case hasGoal && started:
		fmt.Fprintf(os.Stderr,
			"agent: session %s already has a transcript in %s\n\n"+
				"    continue it\n        agent %s\n\n"+
				"    start a new one\n"+
				"        echo '<goal>' | agent $(cat /proc/sys/kernel/random/uuid)\n",
			sessionID, cfg.Session.Dir, sessionID)

		return 1
	case !hasGoal && !started:
		fmt.Fprintf(os.Stderr,
			"agent: session %s has no goal recorded and none on stdin\n\n"+
				"    start it\n        echo '<goal>' | agent %s\n",
			sessionID, sessionID)

		return 1
	case !hasGoal:
		goal = recorded
	}

	var idx *rememberIndex

	if cfg.Remember.EmbeddingModel != "" {
		sp := map[string]any{}
		maps.Copy(sp, cfg.Endpoint.Params)
		maps.Copy(sp, cfg.Remember.SummaryParams)

		summarizer, _ := newClient(&cfg, api, "summary",
			cmp.Or(cfg.Remember.SummaryModel, cfg.Endpoint.Model), sp)

		idx, err = newRememberIndex(
			ctx, &cfg, db, summarizer, api, sessionID, os.Stderr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "agent: remember: %v\n", err)
			return 1
		}

		idx.open = w.first
		store = &rememberStore{inner: store, idx: idx}
		tools = append(tools, functiontool.New(
			"remember", rememberToolDescription, idx.remember))
	}

	opts = append(opts,
		agent.WithTools(tools...),
		agent.WithSession(sessionID, store))

	mode := ""
	if !hasGoal {
		mode = " (resumed)"
	}

	models := "model: " + model.APIModel
	if cfg.Remember.EmbeddingModel != "" {
		models += fmt.Sprintf("  summary: %s  embed: %s",
			cmp.Or(cfg.Remember.SummaryModel, cfg.Endpoint.Model),
			cfg.Remember.EmbeddingModel)
	}

	fmt.Fprintf(os.Stderr,
		"config: %s\nendpoint: %s  context: %d of %d  budget: %d\n%s\n"+
			"session: %s in %s%s\n",
		path, cfg.Endpoint.BaseURL, w.budget, model.ContextWindow,
		cfg.Budget.MaxIterations, models, sessionID, dbPath, mode)

	ag := agent.New(client, opts...)

	began := time.Now()

	resp, err := ag.Chat(ctx, goal)

	elapsed := time.Since(began)

	if err != nil {
		fmt.Fprintf(os.Stderr, "\nagent: %v\n", err)

		if errors.Is(ctx.Err(), context.Canceled) {
			return 130
		}

		return 1
	}

	fmt.Fprintf(os.Stderr,
		"\n[done] %s | %d calls, %d tool calls, %s | tokens in %d / out %d",
		resp.FinishReason, resp.TotalTurns, resp.TotalToolCalls,
		took(elapsed),
		resp.Usage.InputTokens, resp.Usage.OutputTokens)

	if w.sent > 0 {
		fmt.Fprintf(os.Stderr, " | context %d/%d sent", w.sent, w.budget)
	}

	if w.dropped > 0 {
		fmt.Fprintf(os.Stderr, ", %d msgs trimmed", w.dropped)
	}

	if idx != nil {
		total, ready := idx.counts(context.Background())

		fmt.Fprintf(os.Stderr,
			" | remember %d of %d rows embedded, %d summary tokens",
			ready, total, idx.tokens)
	}

	if pace.waits > 0 {
		fmt.Fprintf(os.Stderr, " | waited %d times, %s",
			pace.waits, took(pace.total))
	}

	fmt.Fprintln(os.Stderr)

	if resp.Content != "" {
		fmt.Println(resp.Content)
	}

	if resp.FinishReason == message.FinishReasonMaxIterations {
		return 2
	}

	return 0
}

func main() {
	os.Exit(run())
}
