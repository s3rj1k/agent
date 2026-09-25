// SPDX-License-Identifier: Unlicense

//go:build !recall

package main

import (
	"context"
	"fmt"
)

const recallHelp = "not built in, rebuild with -tags recall to enable it"

const promptSystemRecall = ""

type RecallConfig map[string]any

func recallSection() (string, error) { return "", nil }

func normalizeRecall(cfg *Config, path string) error {
	if len(cfg.Recall) == 0 {
		return nil
	}

	return fmt.Errorf("%s: a recall block is configured but this binary was "+
		"built without the recall feature, so rebuild it with -tags recall "+
		"or delete the block", path)
}

func (*recallSetup) install(context.Context) error { return nil }
