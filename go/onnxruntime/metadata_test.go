package onnxruntime

import (
	"testing"
)

func TestModelMetadata(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	meta, err := sess.ModelMetadata()
	if err != nil {
		t.Fatal(err)
	}
	defer meta.Close()

	_, err = meta.ProducerName()
	if err != nil {
		t.Errorf("ProducerName: %v", err)
	}

	_, err = meta.GraphName()
	if err != nil {
		t.Errorf("GraphName: %v", err)
	}

	_, err = meta.Domain()
	if err != nil {
		t.Errorf("Domain: %v", err)
	}

	_, err = meta.Description()
	if err != nil {
		t.Errorf("Description: %v", err)
	}

	_, err = meta.Version()
	if err != nil {
		t.Errorf("Version: %v", err)
	}

	keys, err := meta.CustomMetadataKeys()
	if err != nil {
		t.Errorf("CustomMetadataKeys: %v", err)
	}
	t.Logf("metadata: keys=%v", keys)

	_, err = meta.LookupCustomMetadata("nonexistent_key")
	if err != nil {
		t.Errorf("LookupCustomMetadata for missing key should not error: %v", err)
	}
}

func TestModelMetadataDoubleClose(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	meta, err := sess.ModelMetadata()
	if err != nil {
		t.Fatal(err)
	}
	meta.Close()
	meta.Close()
}
