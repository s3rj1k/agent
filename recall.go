// SPDX-License-Identifier: Unlicense

//go:build recall

package main

import (
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
	"strings"
	"time"
	"unicode"

	"github.com/joakimcarlsson/ai/llm"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/session"
	"github.com/joakimcarlsson/ai/tool"
	"github.com/joakimcarlsson/ai/tool/functiontool"
	_ "modernc.org/sqlite/vec"
)

const recallHelp = "built in, configured under recall: in the config file"

const (
	defaultMaxFetchBytes = 65536
	defaultTopK          = 5
	maxSearchLimit       = 25
	maxSummaryBytes      = 1000
	recallTries          = 3
	searchHeaderBytes    = 160
)

//go:embed config/prompts/recall-tool.txt
var recallToolDescription string

//go:embed config/prompts/system-recall.txt
var promptSystemRecall string

//go:embed config/prompts/summary.txt
var promptSummary string

//go:embed config/recall.yaml.tmpl
var configRecall string

type RecallConfig struct {
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

type embedder struct {
	api     *endpointAPI
	passage map[string]any
	query   map[string]any
	model   string
	dims    int
}

type recallParams struct {
	ID    *int64 `json:"id,omitempty" desc:"Return one entry in full instead of searching. Use an id from an earlier search result. Ignores query when set."`
	Limit *int   `json:"limit,omitempty" desc:"How many matches to return. Omit to use the configured default."`
	Query string `json:"query,omitempty" desc:"What to look for, described in your own words. Matched by meaning against a short description of every past turn, so a sentence works better than a keyword."`
}

type recallIndex struct {
	db           *sql.DB
	embedder     *embedder
	llm          llm.LLM
	echo         io.Writer
	open         func()
	unsummarized map[int64]int
	unembedded   map[int64]int
	sessionID    string
	prompt       string
	model        string
	topK         int
	maxFetch     int
	tokens       int64
}

type recallPending struct {
	summary sql.Null[string]
	id      int64
	lo, hi  int64
}

type recallSession struct {
	session.Session

	idx *recallIndex
}

type recallStore struct {
	inner session.Store
	idx   *recallIndex
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

func (x *recallIndex) text(msgs []message.Message) string {
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

func (x *recallIndex) maxID(ctx context.Context) int64 {
	var id sql.Null[int64]
	if err := x.db.QueryRowContext(ctx,
		`SELECT MAX(id) FROM messages WHERE session_id = ?`,
		x.sessionID).Scan(&id); err != nil {
		return 0
	}

	return id.V
}

func (x *recallIndex) body(
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

func (x *recallIndex) brief(s string) string {
	if len(s) <= maxSummaryBytes {
		return s
	}

	return strings.ToValidUTF8(s[:maxSummaryBytes], "")
}

func (x *recallIndex) summarize(
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

func (x *recallIndex) embed(
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

	for i, v := range vecs {
		if len(v) == 0 {
			return nil, fmt.Errorf("vector %d came back empty", i)
		}
	}

	return vecs, nil
}

func (x *recallIndex) embedPassage(
	ctx context.Context,
	texts []string,
) ([][]float32, error) {
	return x.embed(ctx, texts, x.embedder.passage)
}

func (x *recallIndex) embedQuery(
	ctx context.Context,
	texts []string,
) ([][]float32, error) {
	return x.embed(ctx, texts, x.embedder.query)
}

func (x *recallIndex) counts(ctx context.Context) (total, ready int) {
	_ = x.db.QueryRowContext(ctx,
		`SELECT count(*), coalesce(sum(vector IS NOT NULL AND model = ?), 0)
		FROM recall WHERE session_id = ?`,
		x.model, x.sessionID).Scan(&total, &ready)

	return total, ready
}

func (x *recallIndex) pending(
	ctx context.Context,
) ([]recallPending, error) {
	rows, err := x.db.QueryContext(ctx,
		`SELECT id, lo_id, hi_id, summary FROM recall
		WHERE session_id = ? AND vector IS NULL ORDER BY id`, x.sessionID)
	if err != nil {
		return nil, err
	}

	defer func() { _ = rows.Close() }()

	var todo []recallPending

	for rows.Next() {
		var p recallPending
		if err := rows.Scan(&p.id, &p.lo, &p.hi, &p.summary); err != nil {
			return nil, err
		}

		todo = append(todo, p)
	}

	return todo, rows.Err()
}

func (x *recallIndex) fill(ctx context.Context) {
	save := context.WithoutCancel(ctx)

	todo, listErr := x.pending(ctx)
	if listErr != nil {
		_, _ = fmt.Fprintf(x.echo, "[recall] %v\n", listErr)

		return
	}

	var (
		ids       []int64
		summaries []string
	)

	for _, p := range todo {
		if x.unembedded[p.id] >= recallTries {
			continue
		}

		if p.summary.Valid {
			ids = append(ids, p.id)
			summaries = append(summaries, p.summary.V)

			continue
		}

		if x.unsummarized[p.id] >= recallTries {
			continue
		}

		body, err := x.body(ctx, p.lo, p.hi)
		if err != nil || body == "" {
			continue
		}

		s, err := x.summarize(ctx, body)
		if err != nil {
			x.unsummarized[p.id]++

			_, _ = fmt.Fprintf(x.echo, "[recall] summary failed: %v\n", err)

			continue
		}

		if _, err := x.db.ExecContext(save,
			`UPDATE recall SET summary = ? WHERE id = ?`, s, p.id); err != nil {
			_, _ = fmt.Fprintf(x.echo, "[recall] %v\n", err)
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
		for _, id := range ids {
			x.unembedded[id]++
		}

		_, _ = fmt.Fprintf(x.echo, "[recall] embedding failed: %v\n", err)

		return
	}

	for i, id := range ids {
		blob, err := binary.Append(nil, binary.LittleEndian, vecs[i])
		if err != nil {
			_, _ = fmt.Fprintf(x.echo, "[recall] %v\n", err)
			continue
		}

		if _, err := x.db.ExecContext(save, `UPDATE recall
			SET summary = ?, vector = ?, dims = ?, model = ? WHERE id = ?`,
			summaries[i], blob, len(vecs[i]), x.model, id); err != nil {
			_, _ = fmt.Fprintf(x.echo, "[recall] %v\n", err)
		}
	}
}

func (x *recallIndex) add(
	ctx context.Context,
	msgs []message.Message,
	lo, hi int64,
) {
	calls, mine := 0, 0

	for i := range msgs {
		for _, c := range msgs[i].ToolCalls() {
			calls++

			if c.Name == "recall" {
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

	if _, err := x.db.ExecContext(ctx, `INSERT INTO recall
		(session_id, lo_id, hi_id, bytes, created_at, model, dims)
		VALUES (?, ?, ?, ?, ?, ?, 0)`,
		x.sessionID, lo, hi, len(body), time.Now().UnixNano(),
		x.model); err != nil {
		_, _ = fmt.Fprintf(x.echo, "[recall] %v\n", err)

		return
	}

	x.fill(ctx)
}

func (x *recallIndex) search(
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
		FROM recall
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

func (x *recallIndex) affordable(fixed, s string) string {
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

func (x *recallIndex) fetch(
	ctx context.Context,
	id int64,
) (string, error) {
	var (
		lo, hi, at int64
		summary    sql.Null[string]
	)

	if err := x.db.QueryRowContext(ctx,
		`SELECT lo_id, hi_id, created_at, summary FROM recall
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

func (x *recallIndex) recall(
	ctx context.Context,
	p recallParams,
) (tool.Response, error) {
	if p.ID != nil {
		out, err := x.fetch(ctx, *p.ID)
		if err != nil {
			return tool.NewTextErrorResponse(
				fmt.Sprintf("recall: %v", err)), nil
		}

		return tool.NewTextResponse(out), nil
	}

	query := strings.TrimSpace(p.Query)
	if query == "" {
		return tool.NewTextErrorResponse(
			`recall: set "query" to search, or "id" to fetch one entry.`), nil
	}

	limit := x.topK
	if p.Limit != nil && *p.Limit > 0 {
		limit = min(*p.Limit, maxSearchLimit)
	}

	out, err := x.search(ctx, query, limit)
	if err != nil {
		return tool.NewTextErrorResponse(
			fmt.Sprintf("recall: %v", err)), nil
	}

	return tool.NewTextResponse(out), nil
}

func (s *recallSession) AddMessages(
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

func (s *recallStore) Exists(
	ctx context.Context,
	id string,
) (bool, error) {
	return s.inner.Exists(ctx, id)
}

func (s *recallStore) Create(
	ctx context.Context,
	id string,
) (session.Session, error) {
	inner, err := s.inner.Create(ctx, id)
	if err != nil {
		return nil, err
	}

	return &recallSession{Session: inner, idx: s.idx}, nil
}

func (s *recallStore) Load(
	ctx context.Context,
	id string,
) (session.Session, error) {
	inner, err := s.inner.Load(ctx, id)
	if err != nil {
		return nil, err
	}

	return &recallSession{Session: inner, idx: s.idx}, nil
}

func (s *recallStore) Delete(ctx context.Context, id string) error {
	if _, err := s.idx.db.ExecContext(ctx,
		`DELETE FROM recall WHERE session_id = ?`, id); err != nil {
		_, _ = fmt.Fprintf(s.idx.echo, "[recall] %v\n", err)
	}

	return s.inner.Delete(ctx, id)
}

func newRecallIndex(
	ctx context.Context,
	cfg *Config,
	db *sql.DB,
	client llm.LLM,
	api *endpointAPI,
	sessionID string,
	out io.Writer,
) (*recallIndex, error) {
	for _, stmt := range []string{
		`CREATE TABLE IF NOT EXISTS recall (
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
		`CREATE INDEX IF NOT EXISTS idx_recall_session
			ON recall(session_id, id)`,
	} {
		if _, err := db.ExecContext(ctx, stmt); err != nil {
			return nil, err
		}
	}

	query := map[string]any{}
	maps.Copy(query, cfg.Recall.EmbeddingParams)
	maps.Copy(query, cfg.Recall.EmbeddingQueryParams)

	return &recallIndex{
		db: db,
		embedder: &embedder{
			api:     api,
			model:   cfg.Recall.EmbeddingModel,
			passage: cfg.Recall.EmbeddingParams,
			query:   query,
			dims:    cfg.Recall.Dimensions,
		},
		llm:          client,
		echo:         out,
		unsummarized: map[int64]int{},
		unembedded:   map[int64]int{},
		sessionID:    sessionID,
		prompt:       cfg.Recall.SummaryPrompt,
		model:        cfg.Recall.EmbeddingModel,
		topK:         cfg.Recall.TopK,
		maxFetch:     cfg.Recall.MaxFetchBytes,
	}, nil
}

func recallSection() (string, error) {
	return renderConfig("recall", configRecall, map[string]string{
		"SummaryPrompt": promptSummary,
	})
}

func normalizeRecall(cfg *Config, path string) error {
	cfg.Recall.Dimensions = max(cfg.Recall.Dimensions, 0)

	if cfg.Recall.TopK < 1 {
		cfg.Recall.TopK = defaultTopK
	}

	cfg.Recall.TopK = min(cfg.Recall.TopK, maxSearchLimit)

	if cfg.Recall.MaxFetchBytes < 1 {
		cfg.Recall.MaxFetchBytes = defaultMaxFetchBytes
	}

	cfg.Recall.MaxFetchBytes = min(cfg.Recall.MaxFetchBytes,
		int(float64(cfg.Endpoint.ContextWindow)*cfg.Endpoint.ContextFraction))

	if cfg.Recall.EmbeddingModel != "" && cfg.Recall.SummaryPrompt == "" {
		return fmt.Errorf("%s: recall.summary_prompt is "+
			"required when recall.embedding_model is set", path)
	}

	return nil
}

func (s *recallSetup) install(ctx context.Context) error {
	if s.cfg.Recall.EmbeddingModel == "" {
		return nil
	}

	sp := map[string]any{}
	maps.Copy(sp, s.cfg.Endpoint.Params)
	maps.Copy(sp, s.cfg.Recall.SummaryParams)

	summary := cmp.Or(s.cfg.Recall.SummaryModel, s.cfg.Endpoint.Model)

	summarizer, _ := newClient(s.cfg, s.api, "summary", summary, sp)

	idx, err := newRecallIndex(
		ctx, s.cfg, s.db, summarizer, s.api, s.sessionID, s.out)
	if err != nil {
		return err
	}

	w := s.win
	idx.open = func() {
		if w.turn == 0 {
			w.openTurn()

			w.opened = true
		}
	}

	out, done := s.out, context.WithoutCancel(ctx)

	s.store = &recallStore{inner: s.store, idx: idx}
	s.tools = append(s.tools, functiontool.New(
		"recall", strings.TrimSpace(recallToolDescription), idx.recall))
	s.banner = fmt.Sprintf("  summary: %s  embed: %s",
		summary, s.cfg.Recall.EmbeddingModel)
	s.report = func() {
		total, ready := idx.counts(done)

		fmt.Fprintf(out, " | recall %d of %d rows embedded, %d summary tokens",
			ready, total, idx.tokens)
	}

	return nil
}
