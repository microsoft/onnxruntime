package onnxruntime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

func closeEPResource(t *testing.T, resource io.Closer) {
	t.Helper()
	if err := resource.Close(); err != nil {
		t.Errorf("close plugin resource: %v", err)
	}
}

func pluginDevice(t *testing.T, registration, path string, gpu bool) *EPDevice {
	t.Helper()
	if err := RegisterExecutionProviderLibrary(registration, path); err != nil {
		t.Fatalf("register plugin %q: %v", path, err)
	}
	t.Cleanup(func() {
		if err := UnregisterExecutionProviderLibrary(registration); err != nil {
			t.Errorf("unregister plugin: %v", err)
		}
	})
	devices, err := GetEPDevices()
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devices {
		matches := d.EPName() == registration && d.Device().Type() == HardwareDeviceTypeCPU
		if gpu {
			matches = strings.Contains(strings.ToLower(d.EPName()), "webgpu") && d.Device().Type() == HardwareDeviceTypeGPU
		}
		if matches {
			version, err := GetVersion()
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("ORT %s API %d, EP %s vendor %s, hardware %s IDs %d/%d, EP metadata %v, hardware metadata %v", version, APIVersion(), d.EPName(), d.EPVendor(), d.Device().Vendor(), d.Device().VendorID(), d.Device().DeviceID(), d.EPMetadata(), d.Device().Metadata())
			return d
		}
	}
	t.Fatalf("plugin %q has no matching device (GPU required: %v)", path, gpu)
	return nil
}

func pluginOptions(t *testing.T, d *EPDevice) *SessionOptions {
	t.Helper()
	o, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { closeEPResource(t, o) })
	for _, err := range []error{
		o.AppendExecutionProviderV2([]*EPDevice{d}, nil),
		o.SetGraphOptimizationLevel(GraphOptimizationLevelDisableAll),
		o.AddConfigEntry("session.disable_cpu_ep_fallback", "1"),
	} {
		if err != nil {
			t.Fatal(err)
		}
	}
	return o
}

func runPluginMul(t *testing.T, s *Session) {
	t.Helper()
	a, err := CreateTensor([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer closeEPResource(t, a)
	b, err := CreateTensor([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	if err != nil {
		t.Fatal(err)
	}
	defer closeEPResource(t, b)
	outputs, err := s.Run(context.Background(), map[string]*Tensor{"A": a, "B": b}, nil)
	defer func() {
		for _, output := range outputs {
			closeEPResource(t, output)
		}
	}()
	if err != nil {
		t.Fatal(err)
	}
	output := outputs["C"]
	if output == nil {
		t.Fatal("Mul output C is missing")
	}
	data, err := TensorData[float32](output)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(data, []float32{10, 40, 90, 160, 250, 360}) || !reflect.DeepEqual(output.Shape(), []int64{2, 3}) {
		t.Fatalf("Mul output %v shape %v", data, output.Shape())
	}
}

func pluginInferenceLifecycle(t *testing.T, path, registration string, gpu bool) {
	t.Helper()
	d := pluginDevice(t, registration, path, gpu)
	o := pluginOptions(t, d)
	if err := o.EnableProfiling(filepath.Join(t.TempDir(), "plugin_profile")); err != nil {
		t.Fatal(err)
	}
	s, err := NewSession(testdataPath("mul_f32.onnx"), o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { closeEPResource(t, s) })
	runPluginMul(t, s)
	profile, err := s.EndProfiling()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(profile)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkPluginProfile(data, d.EPName()); err != nil {
		t.Fatal(err)
	}
	t.Logf("profile %s proves compute on %s", profile, d.EPName())
	closeEPResource(t, s)
	closeEPResource(t, o)
	if err := UnregisterExecutionProviderLibrary(registration); err != nil {
		t.Fatal(err)
	}
	devices, err := GetEPDevices()
	if err != nil {
		t.Fatal(err)
	}
	for _, remaining := range devices {
		if remaining.EPName() == d.EPName() {
			t.Fatalf("unregistered EP %s still discovered", d.EPName())
		}
	}
	stale, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer closeEPResource(t, stale)
	if err := stale.AppendExecutionProviderV2([]*EPDevice{d}, nil); err == nil {
		t.Fatal("stale device token accepted")
	}
	if err := RegisterExecutionProviderLibrary(registration, path); err != nil {
		t.Fatal(err)
	}
	if err := stale.AppendExecutionProviderV2([]*EPDevice{d}, nil); err == nil {
		t.Fatal("stale device token accepted")
	}
	devices, err = GetEPDevices()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, current := range devices {
		if current.EPName() == d.EPName() {
			found = true
		}
	}
	if !found {
		t.Fatal("reregistered plugin was not discovered")
	}
	// The registered library remains for the cleanup registered by pluginDevice.
}

func TestEPWebGPUIntegration(t *testing.T) {
	path := integrationPluginPath(t, "ORT_TEST_WEBGPU_EP_PATH")
	pluginInferenceLifecycle(t, path, "go_webgpu_ep", true)
}

func copyPluginDirectory(t *testing.T, path string) string {
	t.Helper()
	pathToCopy := path
	source := filepath.Dir(path)
	destination := filepath.Join(t.TempDir(), "插件-é")
	err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destination, relative)
		if entry.IsDir() {
			if path != source {
				return filepath.SkipDir
			}
			return os.MkdirAll(target, 0700)
		}
		name := entry.Name()
		if name != filepath.Base(pathToCopy) && !strings.HasSuffix(name, ".dll") && !strings.HasSuffix(name, ".dylib") && !strings.HasSuffix(name, ".so") && !strings.Contains(name, ".so.") {
			return nil
		}
		// Resolve library symlinks so the copy preserves adjacent runtime dependencies.
		info, err := os.Stat(path)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("unsupported plugin dependency %s", path)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, info.Mode().Perm())
	})
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(destination, filepath.Base(path))
}

func TestEPPluginIntegration(t *testing.T) {
	path := integrationPluginPath(t, "ORT_TEST_PLUGIN_EP_PATH")
	pluginInferenceLifecycle(t, copyPluginDirectory(t, path), "go_example_ep", false)
}

func TestEPContextPluginIntegration(t *testing.T) {
	path := integrationPluginPath(t, "ORT_TEST_PLUGIN_EP_PATH")
	d := pluginDevice(t, "go_example_ep", path, false)
	if APIVersion() < 31 {
		t.Fatal("EPContext reader integration requires API 31")
	}
	directory := t.TempDir()
	model := filepath.Join(directory, "context.onnx")
	o := pluginOptions(t, d)
	for key, value := range map[string]string{"ep.context_enable": "1", "ep.context_embed_mode": "0", "ep.context_file_path": model} {
		if err := o.AddConfigEntry(key, value); err != nil {
			t.Fatal(err)
		}
	}
	s, err := NewSession(testdataPath("mul_f32.onnx"), o)
	if err != nil {
		t.Fatal(err)
	}
	closeEPResource(t, s)
	closeEPResource(t, o)
	if _, err := os.Stat(model); err != nil {
		t.Fatalf("generated EPContext model: %v", err)
	}
	payloads, err := filepath.Glob(filepath.Join(directory, "*.ctx"))
	if err != nil {
		t.Fatal(err)
	}
	if len(payloads) != 1 {
		t.Fatalf("expected one external EPContext payload, got %v", payloads)
	}
	payloadPath := payloads[0]
	payload, err := os.ReadFile(payloadPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(payload) != "binary_data" {
		t.Fatalf("example context payload: %q", payload)
	}
	logicalName := filepath.Base(payloadPath)
	if err := os.Remove(payloadPath); err != nil {
		t.Fatal(err)
	}
	var calls atomic.Int32
	reader := func(name string) ([]byte, error) {
		calls.Add(1)
		if name != logicalName {
			return nil, fmt.Errorf("context name %q, expected %q", name, logicalName)
		}
		return payload, nil
	}
	load := pluginOptions(t, d)
	if err := load.SetEPContextDataReader(reader, uint64(len(payload))); err != nil {
		t.Fatal(err)
	}
	loaded, err := NewSession(model, load)
	if err != nil {
		t.Fatalf("load context with reader: %v", err)
	}
	closeEPResource(t, loaded)
	closeEPResource(t, load)
	if calls.Load() == 0 {
		t.Fatal("EPContext reader was not called")
	}
	t.Logf("reader loaded logical payload %q (%d bytes)", logicalName, len(payload))
	// A valid fallback file must never hide a configured reader failure.
	if err := os.WriteFile(payloadPath, payload, 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range contextReaderFailureCases(payload) {
		t.Run(tc.name, func(t *testing.T) {
			opts := pluginOptions(t, d)
			var invoked atomic.Int32
			if err := opts.SetEPContextDataReader(func(name string) ([]byte, error) { invoked.Add(1); return tc.reader(name) }, tc.limit); err != nil {
				t.Fatal(err)
			}
			failed, err := NewSession(model, opts)
			if failed != nil {
				closeEPResource(t, failed)
			}
			requireOrtCode(t, err, tc.code)
			if invoked.Load() == 0 || !strings.Contains(err.Error(), tc.diagnostic) {
				t.Fatalf("reader failure did not propagate: calls %d, %v", invoked.Load(), err)
			}
		})
	}
	t.Run("unsupported_read", func(t *testing.T) {
		opts := pluginOptions(t, d)
		if err := opts.AddConfigEntry("ep.example_ep.test_ep_context_data_support", "0"); err != nil {
			t.Fatal(err)
		}
		var invoked atomic.Int32
		if err := opts.SetEPContextDataReader(func(string) ([]byte, error) { invoked.Add(1); return payload, nil }, uint64(len(payload))); err != nil {
			t.Fatal(err)
		}
		failed, err := NewSession(model, opts)
		if failed != nil {
			closeEPResource(t, failed)
		}
		var native *OrtError
		if !errors.As(err, &native) || native.Code != ErrorCodeFail || !strings.Contains(err.Error(), "does not support the registered EPContext data read callback") || invoked.Load() != 0 {
			t.Fatalf("unsupported READ: calls %d, error %v", invoked.Load(), err)
		}
	})
}
