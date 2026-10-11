package onnxruntime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func cpuEPDevice(t *testing.T) *EPDevice {
	t.Helper()
	devices, err := GetEPDevices()
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devices {
		if d.EPName() == "CPUExecutionProvider" {
			return d
		}
	}
	t.Fatal("CPUExecutionProvider was not discovered")
	return nil
}

func TestEPDevicesCPU(t *testing.T) {
	if APIVersion() < 22 {
		t.Skip("covered by TestEPRequiresAPI22")
	}
	d := cpuEPDevice(t)
	if d.Device().Type() != HardwareDeviceTypeCPU {
		t.Fatalf("CPU hardware type: %v", d.Device().Type())
	}
	t.Logf("EP %s vendor %s, hardware vendor %s IDs %d/%d", d.EPName(), d.EPVendor(), d.Device().Vendor(), d.Device().VendorID(), d.Device().DeviceID())
	for _, maps := range [][2]map[string]string{{d.EPMetadata(), d.EPMetadata()}, {d.EPOptions(), d.EPOptions()}, {d.Device().Metadata(), d.Device().Metadata()}} {
		maps[0]["go-test"] = "changed"
		if _, ok := maps[1]["go-test"]; ok {
			t.Fatal("snapshot maps alias")
		}
	}
	var nilDevice *EPDevice
	if nilDevice.EPName() != "" || nilDevice.EPVendor() != "" || len(nilDevice.EPMetadata()) != 0 || len(nilDevice.EPOptions()) != 0 || len(nilDevice.Device().Metadata()) != 0 {
		t.Fatal("nil device accessors")
	}
}

func checkEPAdd(t *testing.T, opts *SessionOptions, fromBytes bool) {
	t.Helper()
	var s *Session
	var err error
	if fromBytes {
		model, readErr := os.ReadFile(testdataPath("add_f32.onnx"))
		if readErr != nil {
			t.Fatal(readErr)
		}
		s, err = NewSessionFromBytes(model, opts)
	} else {
		s, err = NewSession(testdataPath("add_f32.onnx"), opts)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	a, err := CreateTensor([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := CreateTensor([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	out, err := s.Run(context.Background(), map[string]*Tensor{"A": a, "B": b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, tensor := range out {
			tensor.Close()
		}
	}()
	values, err := TensorData[float32](out["C"])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(values, []float32{11, 22, 33, 44, 55, 66}) || !reflect.DeepEqual(out["C"].Shape(), []int64{2, 3}) {
		t.Fatalf("Add output: %v shape %v", values, out["C"].Shape())
	}
}

func TestEPAppendCPU(t *testing.T) {
	if APIVersion() < 22 {
		t.Skip("covered by TestEPRequiresAPI22")
	}
	for _, options := range []map[string]string{nil, {"use_arena": "0"}} {
		for _, fromBytes := range []bool{false, true} {
			opts, err := NewSessionOptions()
			if err != nil {
				t.Fatal(err)
			}
			if err := opts.AppendExecutionProviderV2([]*EPDevice{cpuEPDevice(t)}, options); err != nil {
				opts.Close()
				t.Fatal(err)
			}
			checkEPAdd(t, opts, fromBytes)
			opts.Close()
		}
	}
}

func TestEPPolicyCPU(t *testing.T) {
	if APIVersion() < 22 {
		t.Skip("covered by TestEPRequiresAPI22")
	}
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	if err := opts.SetEPSelectionPolicy(ExecutionProviderDevicePolicyPreferCPU); err != nil {
		t.Fatal(err)
	}
	checkEPAdd(t, opts, false)
	checkEPAdd(t, opts, true)
}

func TestEPRequiresAPI22(t *testing.T) {
	if APIVersion() >= 22 {
		t.Log("API 22 available; cap-21 process exercises unsupported paths")
		return
	}
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	calls := []func() error{
		func() error { return RegisterExecutionProviderLibrary("", "") },
		func() error { return UnregisterExecutionProviderLibrary("") },
		func() error { _, err := GetEPDevices(); return err },
		func() error { return opts.AppendExecutionProviderV2(nil, nil) },
		func() error { return opts.SetEPSelectionPolicy(-1) },
	}
	for i, call := range calls {
		err := call()
		var native *OrtError
		if !errors.As(err, &native) || native.Code != ErrorCodeNotImplemented || !strings.Contains(native.Msg, "requires API 22") {
			t.Fatalf("call %d: %v", i, err)
		}
	}
	checkEPAdd(t, opts, false)
	checkEPAdd(t, opts, true)
}

func epCounts() (int, int, int64) {
	mu.Lock()
	defer mu.Unlock()
	return protectedOptionsCount, inFlightSessionCreates, sessionCount.Load()
}

func TestEPValidation(t *testing.T) {
	if APIVersion() < 22 {
		t.Skip("covered by TestEPRequiresAPI22")
	}
	d := cpuEPDevice(t)
	other := *d
	other.epName = "OtherExecutionProvider"
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	beforeP, beforeC, beforeS := epCounts()
	for _, devices := range [][]*EPDevice{nil, {nil}, {new(EPDevice)}, {d, d}, {d, &other}} {
		if err := opts.AppendExecutionProviderV2(devices, nil); err == nil {
			t.Fatalf("accepted invalid devices %v", devices)
		}
	}
	for _, options := range []map[string]string{{"bad\x00key": "v"}, {"key": "bad\x00value"}} {
		if err := opts.AppendExecutionProviderV2([]*EPDevice{d}, options); err == nil {
			t.Fatal("accepted NUL option")
		}
	}
	for _, policy := range []ExecutionProviderDevicePolicy{-1, 7} {
		if err := opts.SetEPSelectionPolicy(policy); err == nil {
			t.Fatal("accepted invalid policy")
		}
	}
	for _, args := range [][2]string{{"", "path"}, {"name", ""}, {"bad\x00name", "path"}, {"name", "bad\x00path"}} {
		if err := RegisterExecutionProviderLibrary(args[0], args[1]); err == nil {
			t.Fatal("accepted invalid registration")
		}
	}
	for _, name := range []string{"", "bad\x00name"} {
		if err := UnregisterExecutionProviderLibrary(name); err == nil {
			t.Fatal("accepted invalid unregister")
		}
	}
	// Go validation must not stale the token or protect the options.
	p, c, s := epCounts()
	if p != beforeP || c != beforeC || s != beforeS {
		t.Fatalf("validation changed counts: %d/%d/%d", p, c, s)
	}
	mu.Lock()
	fresh := d.generation == environmentGeneration
	mu.Unlock()
	if !fresh {
		t.Fatal("validation invalidated token")
	}
	var nilOpts *SessionOptions
	if err := nilOpts.AppendExecutionProviderV2([]*EPDevice{d}, nil); err == nil {
		t.Fatal("nil options accepted")
	}
	if err := nilOpts.SetEPSelectionPolicy(ExecutionProviderDevicePolicyDefault); err == nil {
		t.Fatal("nil options accepted")
	}
	closed, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	if err := closed.AppendExecutionProviderV2([]*EPDevice{d}, nil); err == nil {
		t.Fatal("closed options accepted")
	}
	if err := closed.SetEPSelectionPolicy(ExecutionProviderDevicePolicyDefault); err == nil {
		t.Fatal("closed options accepted")
	}
}

func TestEPRegistryLifetime(t *testing.T) {
	if APIVersion() < 22 {
		t.Skip("covered by TestEPRequiresAPI22")
	}
	baseP, baseC, baseS := epCounts()
	d := cpuEPDevice(t)
	name, vendor := d.EPName(), d.EPVendor()
	metadata := d.EPMetadata()
	if err := RegisterExecutionProviderLibrary("go_missing", testdataPath("missing-plugin")); err == nil {
		t.Fatal("missing plugin registered")
	}
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	if err := opts.AppendExecutionProviderV2([]*EPDevice{d}, nil); err == nil {
		t.Fatal("failed register did not stale device")
	}
	if d.EPName() != name || d.EPVendor() != vendor || !reflect.DeepEqual(metadata, d.EPMetadata()) {
		t.Fatal("failed register changed snapshot")
	}
	beforeUnregister := cpuEPDevice(t)
	if err := UnregisterExecutionProviderLibrary("go_unknown"); err == nil {
		t.Fatal("unknown unregister succeeded")
	}
	if err := opts.AppendExecutionProviderV2([]*EPDevice{beforeUnregister}, nil); err == nil {
		t.Fatal("failed unregister did not stale device")
	}
	if err := opts.AppendExecutionProviderV2([]*EPDevice{cpuEPDevice(t)}, nil); err != nil {
		t.Fatal(err)
	}
	clone, err := opts.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	opts.Close()
	opts.Close()
	if err := UnregisterExecutionProviderLibrary("go_unknown"); err == nil || !strings.Contains(err.Error(), "protected options") {
		t.Fatalf("clone protection: %v", err)
	}
	if !clone.successfulV2 || !clone.protected {
		t.Fatal("clone lost selection state")
	}
	checkEPAdd(t, clone, false)
	checkEPAdd(t, clone, true)
	clone.Close()
	clone.Close()
	p, c, s := epCounts()
	if p != baseP || c != baseC || s != baseS {
		t.Fatalf("Close unbalanced counts: %d/%d/%d", p, c, s)
	}
	t.Run("native append failure", func(t *testing.T) {
		o, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		defer o.Close()
		err = o.AppendExecutionProviderV2([]*EPDevice{cpuEPDevice(t)}, map[string]string{strings.Repeat("k", 2048): "v"})
		var native *OrtError
		if !errors.As(err, &native) {
			t.Fatalf("expected native append error, got %v", err)
		}
		if !o.protected || o.successfulV2 {
			t.Fatal("failed append protection/state")
		}
		if err := UnregisterExecutionProviderLibrary("go_unknown"); err == nil || !strings.Contains(err.Error(), "protected options") {
			t.Fatalf("failed append unload guard: %v", err)
		}
	})
	t.Run("reservation", func(t *testing.T) {
		o, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		snapshot, _, err := reserveSessionCreation(o)
		if err != nil {
			o.Close()
			t.Fatal(err)
		}
		defer finishSessionCreation(snapshot, nil)
		// Close and Clone can acquire the source mutex after snapshot capture.
		done := make(chan error, 1)
		go func() {
			copied, err := o.Clone()
			if err == nil {
				err = copied.Close()
			}
			if err == nil {
				err = o.Close()
			}
			done <- err
		}()
		select {
		case err := <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("snapshot retained source lock")
		}
		if err := UnregisterExecutionProviderLibrary("go_unknown"); err == nil || !strings.Contains(err.Error(), "session creations") {
			t.Fatalf("unregister reservation: %v", err)
		}
		if err := RegisterExecutionProviderLibrary("go_missing", "missing-plugin"); err == nil || !strings.Contains(err.Error(), "in progress") {
			t.Fatalf("register reservation: %v", err)
		}
		if err := Shutdown(); err == nil || !strings.Contains(err.Error(), "session creations") {
			t.Fatalf("shutdown reservation: %v", err)
		}
	})
	t.Run("create failures", func(t *testing.T) {
		o, err := NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		defer o.Close()
		if err := o.SetEPSelectionPolicy(ExecutionProviderDevicePolicyPreferCPU); err != nil {
			t.Fatal(err)
		}
		for _, create := range []func() (*Session, error){func() (*Session, error) { return NewSession("missing-model", o) }, func() (*Session, error) { return NewSessionFromBytes([]byte("invalid model"), o) }} {
			session, err := create()
			if err == nil || session != nil {
				t.Fatalf("invalid model creation: %v", err)
			}
			p, c, s := epCounts()
			if p != baseP+1 || c != baseC || s != baseS {
				t.Fatalf("failed create counts: %d/%d/%d", p, c, s)
			}
		}
	})
	t.Run("live sessions", func(t *testing.T) {
		model, err := os.ReadFile(testdataPath("add_f32.onnx"))
		if err != nil {
			t.Fatal(err)
		}
		for _, fromBytes := range []bool{false, true} {
			o, err := NewSessionOptions()
			if err != nil {
				t.Fatal(err)
			}
			if err := o.SetEPSelectionPolicy(ExecutionProviderDevicePolicyPreferCPU); err != nil {
				o.Close()
				t.Fatal(err)
			}
			clone, err := o.Clone()
			if err != nil {
				o.Close()
				t.Fatal(err)
			}
			if !clone.policyAttempted {
				t.Fatal("clone lost policy state")
			}
			var session *Session
			if fromBytes {
				session, err = NewSessionFromBytes(model, clone)
			} else {
				session, err = NewSession(testdataPath("add_f32.onnx"), clone)
			}
			o.Close()
			clone.Close()
			if err != nil {
				t.Fatal(err)
			}
			err = UnregisterExecutionProviderLibrary("go_unknown")
			if err == nil || !strings.Contains(err.Error(), "1 sessions") {
				session.Close()
				t.Fatalf("session unload guard: %v", err)
			}
			session.Close()
			session.Close()
			p, c, s := epCounts()
			if p != baseP || c != baseC || s != baseS {
				t.Fatalf("session Close counts: %d/%d/%d", p, c, s)
			}
		}
	})

	p, c, s = epCounts()
	if p != baseP || c != baseC || s != baseS {
		t.Fatalf("lifetime counts: %d/%d/%d", p, c, s)
	}
}

func TestEPIntrospectionFailure(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	if APIVersion() >= 22 {
		if err := opts.SetEPSelectionPolicy(ExecutionProviderDevicePolicyPreferCPU); err != nil {
			t.Fatal(err)
		}
	}
	baseP, baseC, baseS := epCounts()
	for _, failInput := range []bool{true, false} {
		session, err := NewSession(testdataPath("add_f32.onnx"), opts)
		if err != nil {
			t.Fatal(err)
		}
		snapshot, _, err := reserveSessionCreation(opts)
		if err != nil {
			session.Close()
			t.Fatal(err)
		}
		sentinel := errors.New("forced IO introspection error")
		result, err := finalizeSessionWithIO(session.handle, func(input bool) ([]IOInfo, error) {
			if input == failInput {
				return nil, sentinel
			}
			return introspectIO(session.handle, input)
		})
		// finalizeSessionWithIO consumed the native session; Close frees the old name cache.
		session.handle = nil
		session.Close()
		finishSessionCreation(snapshot, nil)
		if result != nil || !errors.Is(err, sentinel) {
			t.Fatalf("introspection error: %v", err)
		}
		p, c, s := epCounts()
		if p != baseP || c != baseC || s != baseS {
			t.Fatalf("introspection counts: %d/%d/%d", p, c, s)
		}
	}
}

func TestEPShutdown(t *testing.T) {
	const childMode = epShutdownEnv
	if os.Getenv(childMode) != "1" {
		out, err := runRuntimeChild(t, ".", testRuntimePath, childMode, "^TestEPShutdown$")
		if err != nil || !strings.Contains(string(out), "--- PASS: TestEPShutdown") {
			t.Fatalf("shutdown child: %v\n%s", err, out)
		}
		return
	}
	if APIVersion() < 22 {
		t.Skip("API 22 unavailable")
	}
	d := cpuEPDevice(t)
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	if err := opts.SetEPSelectionPolicy(ExecutionProviderDevicePolicyPreferCPU); err != nil {
		opts.Close()
		t.Fatal(err)
	}
	clone, err := opts.Clone()
	if err != nil {
		opts.Close()
		t.Fatal(err)
	}
	opts.Close()
	if err := Shutdown(); err == nil {
		clone.Close()
		t.Fatal("Shutdown ignored protected clone")
	}
	clone.Close()
	clone.Close()
	if err := Shutdown(); err != nil {
		t.Fatal(err)
	}
	if d.EPName() != "CPUExecutionProvider" {
		t.Fatal("shutdown erased copied metadata")
	}
	if _, err := GetEPDevices(); !errors.Is(err, errShutdown) {
		t.Fatalf("discovery after shutdown: %v", err)
	}
}

func checkPluginProfile(data []byte, provider string) error {
	var events []struct {
		Name     string `json:"name"`
		Category string `json:"cat"`
		Args     struct {
			Provider string `json:"provider"`
			OpName   string `json:"op_name"`
		} `json:"args"`
	}
	if err := json.Unmarshal(data, &events); err != nil {
		return fmt.Errorf("decode plugin profile: %w", err)
	}
	found := false
	for _, event := range events {
		if event.Category != "Node" || !strings.HasSuffix(event.Name, "_kernel_time") {
			continue
		}
		if event.Args.OpName == "Mul" && event.Args.Provider == "CPUExecutionProvider" {
			return fmt.Errorf("Mul ran on CPUExecutionProvider")
		}
		if event.Args.Provider == provider {
			found = true
		}
	}
	if !found {
		return fmt.Errorf("profile has no compute event for selected EP %s", provider)
	}
	return nil
}

func TestEPPluginProfile(t *testing.T) {
	for _, tc := range []struct {
		name, data string
		valid      bool
	}{
		{"plugin", `[{"cat":"Node","name":"Mul_kernel_time","args":{"provider":"go_example_ep","op_name":"Mul"}}]`, true},
		{"omitted_append", `[{"cat":"Node","name":"Mul_kernel_time","args":{"provider":"CPUExecutionProvider","op_name":"Mul"}}]`, false},
		{"non_compute", `[{"cat":"Session","name":"session_initialization","args":{"provider":"go_example_ep"}}]`, false},
		{"cpu_mul", `[{"cat":"Node","name":"Mul_kernel_time","args":{"provider":"CPUExecutionProvider","op_name":"Mul"}},{"cat":"Node","name":"Other_kernel_time","args":{"provider":"go_example_ep"}}]`, false},
		{"invalid", `{`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := checkPluginProfile([]byte(tc.data), "go_example_ep"); (err == nil) != tc.valid {
				t.Fatalf("profile result: %v", err)
			}
		})
	}
}

func TestEPProfileRequiresAppend(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer closeEPResource(t, opts)
	if err := opts.EnableProfiling(filepath.Join(t.TempDir(), "cpu_profile")); err != nil {
		t.Fatal(err)
	}
	// Omit V2 append deliberately: correct values alone cannot prove plugin execution.
	session, err := NewSession(testdataPath("mul_f32.onnx"), opts)
	if err != nil {
		t.Fatal(err)
	}
	defer closeEPResource(t, session)
	runPluginMul(t, session)
	profile, err := session.EndProfiling()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(profile)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkPluginProfile(data, "go_example_ep"); err == nil || !strings.Contains(err.Error(), "Mul ran on CPUExecutionProvider") {
		t.Fatalf("omitted append did not fail provider assertion: %v", err)
	}
}
