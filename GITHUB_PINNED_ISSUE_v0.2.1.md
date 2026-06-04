# What Project Synapse Is, Why It Exists, and How To Use It

Project Synapse is a public research prototype for studying **resource-conditioned hypernetwork adaptation**.

The core idea is simple:

1. generate lightweight task parameters from task identity and runtime context,
2. route them through a modular substrate,
3. reuse previously successful parameter states through a ledger-like memory.

## Why This Repository Exists

Many adaptation methods are static once configured. Synapse explores whether adaptation can become more context-aware at runtime, especially for edge or resource-constrained settings.

## What You Can Use Today

1. a reproducible public quick-start on a real dataset,
2. a resource-aware demo path,
3. an adaptation analysis script for comparing generated weights across contexts,
4. a feature extraction utility for ImageFolder-style datasets.

## What This Repository Is Not Claiming

1. a new fundamental learning paradigm,
2. a production-ready framework,
3. a completed benchmark study against all standard baselines.

## Best Way To Read The Project

Treat Synapse as:

1. a research artifact,
2. a benchmark starting point,
3. a useful base for extending into fixed-head, adapter, or LoRA comparisons.
