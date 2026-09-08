package onnxruntime

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func findQwen3Model() string {
	if p := os.Getenv("QWEN3_ONNX_PATH"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	glob := filepath.Join(home, ".cache/huggingface/hub/models--onnx-community--Qwen3-Embedding-0.6B-ONNX/snapshots/*/onnx/model_int8.onnx")
	matches, _ := filepath.Glob(glob)
	if len(matches) > 0 {
		return matches[0]
	}
	return ""
}

func TestQwen3EmbeddingIntrospection(t *testing.T) {
	modelPath := findQwen3Model()
	if modelPath == "" {
		t.Skip("Qwen3-Embedding model not found; set QWEN3_ONNX_PATH or download via huggingface-cli")
	}

	sess, err := NewSession(modelPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	inputs := sess.Inputs()
	if len(inputs) != 59 {
		t.Fatalf("expected 59 inputs, got %d", len(inputs))
	}

	if inputs[0].Name != "input_ids" || inputs[0].DataType != TensorElementDataTypeInt64 {
		t.Errorf("input[0]: expected input_ids int64, got %s %s", inputs[0].Name, inputs[0].DataType)
	}
	if inputs[1].Name != "attention_mask" || inputs[1].DataType != TensorElementDataTypeInt64 {
		t.Errorf("input[1]: expected attention_mask int64, got %s %s", inputs[1].Name, inputs[1].DataType)
	}
	if inputs[2].Name != "position_ids" || inputs[2].DataType != TensorElementDataTypeInt64 {
		t.Errorf("input[2]: expected position_ids int64, got %s %s", inputs[2].Name, inputs[2].DataType)
	}

	for i := 3; i < 59; i++ {
		in := inputs[i]
		if !strings.HasPrefix(in.Name, "past_key_values.") {
			t.Errorf("input[%d]: expected past_key_values.*, got %s", i, in.Name)
		}
		if in.DataType != TensorElementDataTypeFloat32 {
			t.Errorf("input %s: expected Float32, got %s", in.Name, in.DataType)
		}
	}

	outputs := sess.Outputs()
	if len(outputs) != 57 {
		t.Fatalf("expected 57 outputs, got %d", len(outputs))
	}
	if outputs[0].Name != "last_hidden_state" || outputs[0].DataType != TensorElementDataTypeFloat32 {
		t.Errorf("output[0]: expected last_hidden_state float32, got %s %s", outputs[0].Name, outputs[0].DataType)
	}
}

func TestQwen3EmbeddingInference(t *testing.T) {
	modelPath := findQwen3Model()
	if modelPath == "" {
		t.Skip("Qwen3-Embedding model not found; set QWEN3_ONNX_PATH or download via huggingface-cli")
	}

	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	opts.SetIntraOpNumThreads(4)
	opts.SetGraphOptimizationLevel(GraphOptimizationLevelAll)

	sess, err := NewSession(modelPath, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	seqLen := int64(5)
	tokenIDs := []int64{151644, 8948, 198, 2610, 525} // <|im_start|> system \n You are
	attentionMask := make([]int64, seqLen)
	positionIDs := make([]int64, seqLen)
	for i := int64(0); i < seqLen; i++ {
		attentionMask[i] = 1
		positionIDs[i] = i
	}

	inputMap := make(map[string]*Tensor)
	var tensorsToClose []*Tensor

	inputIDs, err := CreateTensor[int64]([]int64{1, seqLen}, tokenIDs)
	if err != nil {
		t.Fatal(err)
	}
	tensorsToClose = append(tensorsToClose, inputIDs)
	inputMap["input_ids"] = inputIDs

	mask, err := CreateTensor[int64]([]int64{1, seqLen}, attentionMask)
	if err != nil {
		t.Fatal(err)
	}
	tensorsToClose = append(tensorsToClose, mask)
	inputMap["attention_mask"] = mask

	pos, err := CreateTensor[int64]([]int64{1, seqLen}, positionIDs)
	if err != nil {
		t.Fatal(err)
	}
	tensorsToClose = append(tensorsToClose, pos)
	inputMap["position_ids"] = pos

	for i := 0; i < 28; i++ {
		for _, role := range []string{"key", "value"} {
			name := fmt.Sprintf("past_key_values.%d.%s", i, role)
			kv, err := CreateTensor[float32]([]int64{1, 8, 0, 128}, []float32{})
			if err != nil {
				t.Fatalf("create %s: %v", name, err)
			}
			tensorsToClose = append(tensorsToClose, kv)
			inputMap[name] = kv
		}
	}

	defer func() {
		for _, tensor := range tensorsToClose {
			tensor.Close()
		}
	}()

	results, err := sess.Run(context.Background(), inputMap, []string{"last_hidden_state"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, r := range results {
			r.Close()
		}
	}()

	out, ok := results["last_hidden_state"]
	if !ok {
		t.Fatal("last_hidden_state not in output")
	}

	shape := out.Shape()
	if len(shape) != 3 {
		t.Fatalf("expected 3D output, got %dD: %v", len(shape), shape)
	}
	if shape[0] != 1 {
		t.Errorf("expected batch=1, got %d", shape[0])
	}
	if shape[1] != seqLen {
		t.Errorf("expected seq=%d, got %d", seqLen, shape[1])
	}
	if shape[2] != 1024 {
		t.Errorf("expected hidden=1024, got %d", shape[2])
	}

	data, err := TensorData[float32](out)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != int(seqLen)*1024 {
		t.Errorf("expected %d elements, got %d", seqLen*1024, len(data))
	}

	allZero := true
	for _, v := range data[:100] {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("output is all zeros — model may not have run correctly")
	}

	t.Logf("Qwen3-Embedding inference OK: output shape %v, first values: [%.4f, %.4f, %.4f, ...]",
		shape, data[0], data[1], data[2])
}
