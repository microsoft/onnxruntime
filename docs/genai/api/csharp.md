---
title: C# API
description: C# API reference for ONNX Runtime GenAI
has_children: false
parent: API docs
grand_parent: Generative AI (Preview)
nav_order: 2
---

# ONNX Runtime GenAI C# API
{: .no_toc }

* TOC placeholder
{:toc}

## Overview

## Model class

### Constructor

```csharp
public Model(string modelPath, DeviceType deviceType)
```

### Generate method

```csharp
public Sequences Generate(GeneratorParams generatorParams)
```

## Tokenizer class

### Constructor

```csharp
public Tokenizer(Model model)
```

### Encode method

```csharp
public Sequences Encode(string str)
```

### Encode batch method

```csharp
public Sequences EncodeBatch(string[] strings)
```

### Decode method

```csharp
public string Decode(ReadOnlySpan<int> sequence)
```

### Decode batch method

```csharp
public string[] DecodeBatch(Sequences sequences)
```

### Create stream method

```csharp
public TokenizerStream CreateStream()
```

## TokenizerStream class

### Decode method

```csharp
public string Decode(int token)
```

## GeneratorParams class

### Constructor

```csharp
public GeneratorParams(Model model)
```

### Set search option (double)

```csharp
public void SetSearchOption(string searchOption, double value)
```

### Set search option (bool) method

```csharp
public void SetSearchOption(string searchOption, bool value)
```

### Set input ids method

```csharp
public void SetInputIDs(ReadOnlySpan<int> inputIDs, ulong sequenceLength, ulong batchSize)
```

### Set input sequences method

```csharp
public void SetInputSequences(Sequences sequences)
```





## Generator class

### Constructor

```csharp
public Generator(Model model, GeneratorParams generatorParams)
```

### Is done method

```csharp
public bool IsDone()
```

### Compute logits

```csharp
public void ComputeLogits()
```

### Generate next token method

```csharp
public void GenerateNextTokenTop()
```


## Sequences class

### Num sequences member

```csharp
public ulong NumSequences { get { return _numSequences; } }
```

### [] operator

```csharp
public ReadOnlySpan<int> this[ulong sequenceIndex]
```

