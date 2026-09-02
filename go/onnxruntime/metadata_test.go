package onnxruntime

import (
	"sync"
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

func TestModelMetadataUseAfterClose(t *testing.T) {
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

	_, err = meta.ProducerName()
	if err == nil {
		t.Fatal("expected error calling ProducerName after Close")
	}
}

func TestModelMetadataGraphDescription(t *testing.T) {
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

	_, err = meta.GraphDescription()
	if err != nil {
		t.Errorf("GraphDescription: %v", err)
	}
}

func TestModelMetadataConcurrentClose(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	meta, err := sess.ModelMetadata()
	if err != nil {
		t.Fatal(err)
	}

	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		for range 100 {
			_, _ = meta.ProducerName()
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		_ = meta.Close()
	}()
	close(start)
	wg.Wait()

	if _, err := meta.Version(); err == nil {
		t.Fatal("expected error after concurrent Close")
	}
}

func TestNilModelMetadata(t *testing.T) {
	var meta *ModelMetadata
	if err := meta.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := meta.ProducerName(); err == nil {
		t.Fatal("expected error using nil metadata")
	}
}

func TestModelMetadataRejectsNULKey(t *testing.T) {
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

	if _, err := meta.LookupCustomMetadata("key\x00ignored"); err == nil {
		t.Fatal("expected error for NUL in metadata key")
	}
}
