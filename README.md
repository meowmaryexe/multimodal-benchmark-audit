# Multimodal Benchmark Auditing (Attempt)

## Goal
Test whether multimodal benchmarks actually require images, or whether models rely on textual shortcuts.

## Core Idea
Run counterfactual experiments:
- Normal (image + question)
- No image (blank image)
- Image swap (later)

Measure how much performance actually depends on the image.

## Current Stage
Day 1: Dead-end test using ChartQA + Qwen2-VL (30 samples)

## Decision Rule
If performance does not significantly depend on images → strong signal → continue  
If results are trivial or noisy → reconsider direction