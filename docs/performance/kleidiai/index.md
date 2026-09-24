---
layout: default
title: Arm® KleidiAI™ software
description: Arm® KleidiAI™ software micro-kernel compatibility in ONNX Runtime
parent: Performance
nav_order: 7
toc: false
redirect_from:
  - /docs/reference/kleidiai/
---
<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->

<div class="kleidiai-page" markdown="1">

# Arm® KleidiAI™ software compatibility in ONNX Runtime

This page outlines support details for the [Arm KleidiAI software](https://www.arm.com/markets/artificial-intelligence/software/kleidi) micro-kernel library. Micro-kernels are integrated into ONNX Runtime via Microsoft Linear Algebra Subprograms (MLAS) to support CPU-based inference acceleration on Arm-based CPUs.
{: .kai-lede }

Arm, Kleidi, KleidiAI, KleidiCV and Kleidi Libraries are registered trademarks or trademarks of Arm Limited (or its subsidiaries or affiliates) in the US and/or elsewhere.
{: .muted .trademark-notice }

**KleidiAI tagged release:** [`v1.31.0`](https://github.com/ARM-software/kleidiai/tree/v1.31.0) · **ONNX Runtime release:** [`v1.30.0`](https://github.com/microsoft/onnxruntime/tree/v1.30.0) · **ONNX Runtime KleidiAI pin:** [`v1.20.0`](https://github.com/ARM-software/kleidiai/tree/v1.20.0) · **Last updated date:** `2026-09-24`
{: .kai-meta }

> Eligible ONNX operators are a conservative set traced directly from integrated MLAS paths. Actual acceleration also depends on datatype, shape, CPU features, packing, and runtime configuration.
{: .kai-note }

## Compatibility table

<link rel="stylesheet" href="kleidiai.css">
<div id="kleidiai-app"><p role="status">Loading compatibility data…</p></div>
<script src="kleidiai.js" defer></script>

</div>
