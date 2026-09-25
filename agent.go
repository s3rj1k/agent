// SPDX-License-Identifier: Unlicense

package main

import (
	"bytes"
	"cmp"
	"context"
	"database/sql"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"text/template"
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
	yaml "go.yaml.in/yaml/v3"
	_ "modernc.org/sqlite"
)

const (
	functionType     = "function"
	maxBackoff       = 5 * time.Minute
	maxBackoffShift  = 4
	maxTokenizerCost = 1 << 28
	retryBackoff     = 30 * time.Second
	textType         = "text"
	typeKey          = "type"
)

//go:embed config/usage.txt
var defaultUsage string

//go:embed config/agent.yaml.tmpl
var configTemplate string

//go:embed config/prompts/system-head.txt
var promptSystemHead string

//go:embed config/prompts/system-tail.txt
var promptSystemTail string

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

type SessionConfig struct {
	Dir string `yaml:"dir"`
}

type Config struct {
	SystemPrompt string         `yaml:"system_prompt"`
	Usage        string         `yaml:"usage"`
	Session      SessionConfig  `yaml:"session"`
	Budget       BudgetConfig   `yaml:"budget"`
	Shell        ShellConfig    `yaml:"shell"`
	Recall       RecallConfig   `yaml:"recall"`
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

type recallSetup struct {
	cfg       *Config
	db        *sql.DB
	api       *endpointAPI
	win       *window
	out       io.Writer
	store     session.Store
	report    func()
	sessionID string
	banner    string
	tools     []tool.BaseTool
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
		Usage:   defaultUsage,
		Session: SessionConfig{Dir: ".agent"},
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

	cfg.Usage = cmp.Or(cfg.Usage, defaults.Usage)

	cfg.Session.Dir = cmp.Or(cfg.Session.Dir, defaults.Session.Dir)

	if raw := cfg.Endpoint.APIKey; envReference.MatchString(raw) {
		if cfg.Endpoint.APIKey = os.ExpandEnv(raw); cfg.Endpoint.APIKey == "" {
			return cfg, path, fmt.Errorf(
				"%s: endpoint.api_key is %q but that expands to nothing, so "+
					"give it a value, set the key literally, or use \"\" for "+
					"an endpoint that wants no auth", path, raw)
		}
	}

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

	if err := normalizeShell(&cfg, path); err != nil {
		return cfg, path, err
	}

	if err := normalizeRecall(&cfg, path); err != nil {
		return cfg, path, err
	}

	return cfg, path, nil
}

func printHelp(out io.Writer, text string) {
	_, _ = fmt.Fprint(out, text)

	if path, err := configPath(); err != nil {
		_, _ = fmt.Fprintf(out,
			"\nThe config location could not be resolved, %v\n", err)
	} else {
		state := "does not exist yet, so run agent --init"
		if _, err := os.Stat(path); err == nil {
			state = "already exists"
		}

		_, _ = fmt.Fprintf(out, "\nConfig file\n    %s (%s)\n", path, state)
	}

	_, _ = fmt.Fprintf(out, "\nRecall, the search over your own past turns\n"+
		"    %s\n", recallHelp)
}

func indent(n int, s string) string {
	pad := strings.Repeat(" ", n)
	lines := strings.Split(strings.TrimSpace(s), "\n")

	for i, l := range lines {
		if l != "" {
			lines[i] = pad + l
		}
	}

	return strings.Join(lines, "\n")
}

func renderConfig(name, src string, vars map[string]string) (string, error) {
	t, err := template.New(name).Delims("[[", "]]").
		Funcs(template.FuncMap{"indent": indent}).Parse(src)
	if err != nil {
		return "", fmt.Errorf("%s template: %w", name, err)
	}

	var b strings.Builder
	if err := t.Execute(&b, vars); err != nil {
		return "", fmt.Errorf("%s template: %w", name, err)
	}

	return strings.TrimSpace(b.String()) + "\n", nil
}

func systemPrompt() string {
	parts := make([]string, 0, 3)

	for _, p := range []string{
		promptSystemHead, promptSystemRecall, promptSystemTail,
	} {
		if t := strings.TrimSpace(p); t != "" {
			parts = append(parts, t)
		}
	}

	return strings.Join(parts, "\n\n")
}

func exampleConfig() (string, error) {
	shell, err := shellSection()
	if err != nil {
		return "", err
	}

	recall, err := recallSection()
	if err != nil {
		return "", err
	}

	return renderConfig("config", configTemplate, map[string]string{
		"SystemPrompt": systemPrompt(),
		"Shell":        strings.TrimSpace(shell),
		"Recall":       strings.TrimSpace(recall),
	})
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

	content, err := exampleConfig()
	if err != nil {
		_, _ = fmt.Fprintf(out, "agent: %v\n", err)

		return 1
	}

	if err = os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		_, _ = fmt.Fprintf(out, "agent: %v\n", err)

		return 1
	}

	if err = os.WriteFile(path, []byte(content), 0o600); err != nil {
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

func run() int {
	args := os.Args[1:]

	cfg, path, cfgErr := LoadConfig()

	usage := cmp.Or(cfg.Usage, defaultUsage)

	if len(args) == 1 {
		switch {
		case slices.Contains([]string{"-h", "--help", "help"}, args[0]):
			printHelp(os.Stderr, usage)

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

	if cfgErr != nil {
		fmt.Fprintf(os.Stderr, "agent: %v\n", cfgErr)
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

	tools := []tool.BaseTool{newExecTool(&cfg, os.Stderr)}

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

	rc := &recallSetup{
		cfg:       &cfg,
		db:        db,
		api:       api,
		win:       w,
		out:       os.Stderr,
		store:     store,
		tools:     tools,
		sessionID: sessionID,
	}

	if err = rc.install(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "agent: recall: %v\n", err)
		return 1
	}

	opts = append(opts,
		agent.WithTools(rc.tools...),
		agent.WithSession(sessionID, rc.store))

	mode := ""
	if !hasGoal {
		mode = " (resumed)"
	}

	models := "model: " + model.APIModel + rc.banner

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

	if rc.report != nil {
		rc.report()
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
