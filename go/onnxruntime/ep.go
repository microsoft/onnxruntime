package onnxruntime

/*
#include "cshim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math"
	"unsafe"
)

// HardwareDeviceType identifies a device's processing unit.
type HardwareDeviceType int

const (
	HardwareDeviceTypeCPU HardwareDeviceType = iota
	HardwareDeviceTypeGPU
	HardwareDeviceTypeNPU
)

// ExecutionProviderDevicePolicy controls automatic execution provider selection.
type ExecutionProviderDevicePolicy int

const (
	ExecutionProviderDevicePolicyDefault ExecutionProviderDevicePolicy = iota
	ExecutionProviderDevicePolicyPreferCPU
	ExecutionProviderDevicePolicyPreferNPU
	ExecutionProviderDevicePolicyPreferGPU
	ExecutionProviderDevicePolicyMaxPerformance
	ExecutionProviderDevicePolicyMaxEfficiency
	ExecutionProviderDevicePolicyMinOverallPower
)

// EPDevice is an immutable snapshot of a discovered provider/device pair.
// Registry mutations invalidate selection tokens, but copied attributes remain readable.
// Only GetEPDevices creates valid selection tokens. Accessors are safe for concurrent use.
type EPDevice struct {
	handle                *C.OrtEpDevice
	generation            uint64
	epName, epVendor      string
	epMetadata, epOptions map[string]string
	device                HardwareDevice
}

// HardwareDevice contains copied hardware attributes. A zero value is not a selection token.
// VendorID and DeviceID identify hardware type, not a unique physical device.
type HardwareDevice struct {
	deviceType         HardwareDeviceType
	vendorID, deviceID uint32
	vendor             string
	metadata           map[string]string
}

// EPName returns the provider name.
func (d *EPDevice) EPName() string {
	if d == nil {
		return ""
	}
	return d.epName
}

// EPVendor returns the provider vendor.
func (d *EPDevice) EPVendor() string {
	if d == nil {
		return ""
	}
	return d.epVendor
}

// EPMetadata returns a fresh copy of provider metadata.
func (d *EPDevice) EPMetadata() map[string]string {
	if d == nil {
		return nil
	}
	return copyEPMap(d.epMetadata)
}

// EPOptions returns a fresh copy of provider options.
func (d *EPDevice) EPOptions() map[string]string {
	if d == nil {
		return nil
	}
	return copyEPMap(d.epOptions)
}

// Device returns the hardware snapshot.
func (d *EPDevice) Device() HardwareDevice {
	if d == nil {
		return HardwareDevice{}
	}
	return d.device
}

// Type returns the processing unit type.
func (d HardwareDevice) Type() HardwareDeviceType { return d.deviceType }

// VendorID returns the hardware vendor identifier.
func (d HardwareDevice) VendorID() uint32 { return d.vendorID }

// Vendor returns the hardware vendor name.
func (d HardwareDevice) Vendor() string { return d.vendor }

// DeviceID returns the hardware type identifier.
func (d HardwareDevice) DeviceID() uint32 { return d.deviceID }

// Metadata returns a fresh copy of hardware metadata.
func (d HardwareDevice) Metadata() map[string]string { return copyEPMap(d.metadata) }

func copyEPMap(source map[string]string) map[string]string {
	out := make(map[string]string, len(source))
	for k, v := range source {
		out[k] = v
	}
	return out
}

// requirePluginAPI must run under package mu after the initialization check.
func requirePluginAPI(operation string) error {
	if apiVersion < 22 {
		return wrapErr(operation, &OrtError{Code: ErrorCodeNotImplemented, Msg: fmt.Sprintf("%s requires API 22; negotiated API %d (function unavailable)", operation, apiVersion)})
	}
	return nil
}

func validateRegistrationName(name string) error {
	if name == "" {
		return fmt.Errorf("ort: execution provider registration name is empty")
	}
	return rejectNUL(name, "execution provider registration name")
}

// RegisterExecutionProviderLibrary loads a plugin library. Requires API 22.
// Call during setup, without concurrent inference. Attempts invalidate all existing device tokens.
func RegisterExecutionProviderLibrary(registrationName, libraryPath string) error {
	mu.Lock()
	defer mu.Unlock()
	if err := checkInit(); err != nil {
		return err
	}
	if err := requirePluginAPI("RegisterExecutionProviderLibrary"); err != nil {
		return err
	}
	if err := validateRegistrationName(registrationName); err != nil {
		return err
	}
	if libraryPath == "" {
		return fmt.Errorf("ort: execution provider library path is empty")
	}
	path, free, err := ortPath(libraryPath)
	if err != nil {
		return err
	}
	defer free()
	if inFlightSessionCreates != 0 {
		return fmt.Errorf("ort: register provider: session creation is in progress")
	}
	name := C.CString(registrationName)
	defer C.free(unsafe.Pointer(name))
	environmentGeneration++
	return wrapErr("register execution provider library", checkStatus(C.ort_RegisterExecutionProviderLibrary(env, name, path)))
}

// UnregisterExecutionProviderLibrary unloads a plugin. Requires API 22.
// Close outputs, inputs, bindings, sessions and configured options first.
// Call during teardown, without concurrent inference. Attempts invalidate all device tokens.
func UnregisterExecutionProviderLibrary(registrationName string) error {
	mu.Lock()
	defer mu.Unlock()
	if err := checkInit(); err != nil {
		return err
	}
	if err := requirePluginAPI("UnregisterExecutionProviderLibrary"); err != nil {
		return err
	}
	if err := validateRegistrationName(registrationName); err != nil {
		return err
	}
	if protectedOptionsCount != 0 || sessionCount.Load() != 0 || inFlightSessionCreates != 0 {
		return fmt.Errorf("ort: unregister provider: %d protected options, %d sessions, %d session creations still live", protectedOptionsCount, sessionCount.Load(), inFlightSessionCreates)
	}
	name := C.CString(registrationName)
	defer C.free(unsafe.Pointer(name))
	environmentGeneration++
	return wrapErr("unregister execution provider library", checkStatus(C.ort_UnregisterExecutionProviderLibrary(env, name)))
}

// GetEPDevices discovers provider/device pairs and copies their attributes. Requires API 22.
// Rediscover after every register or unregister attempt before selecting devices.
func GetEPDevices() ([]*EPDevice, error) {
	mu.Lock()
	defer mu.Unlock()
	if err := checkInit(); err != nil {
		return nil, err
	}
	if err := requirePluginAPI("GetEpDevices"); err != nil {
		return nil, err
	}
	var array **C.OrtEpDevice
	var count C.size_t
	if err := checkStatus(C.ort_GetEpDevices(env, &array, &count)); err != nil {
		return nil, wrapErr("get EP devices", err)
	}
	if err := checkEPArray(unsafe.Pointer(array), count); err != nil {
		return nil, err
	}
	devices := make([]*EPDevice, 0, int(count))
	for _, ptr := range unsafe.Slice(array, int(count)) {
		if ptr == nil {
			return nil, fmt.Errorf("ort: get EP devices: nil device")
		}
		d, err := snapshotEPDevice(ptr)
		if err != nil {
			return nil, wrapErr("get EP devices", err)
		}
		devices = append(devices, d)
	}
	return devices, nil
}

func checkEPArray(array unsafe.Pointer, count C.size_t) error {
	if uint64(count) > uint64(math.MaxInt)/uint64(unsafe.Sizeof(uintptr(0))) {
		return fmt.Errorf("ort: device array count %d exceeds addressable range", uint64(count))
	}
	if count != 0 && array == nil {
		return fmt.Errorf("ort: nil device array for count %d", uint64(count))
	}
	return nil
}

func snapshotEPMap(pairs *C.OrtKeyValuePairs) (map[string]string, error) {
	if pairs == nil {
		return nil, nil
	}
	var keys, values **C.char
	var count C.size_t
	if err := checkStatus(C.ort_GetKeyValuePairs(pairs, &keys, &values, &count)); err != nil {
		return nil, err
	}
	if err := checkEPArray(unsafe.Pointer(keys), count); err != nil {
		return nil, err
	}
	if err := checkEPArray(unsafe.Pointer(values), count); err != nil {
		return nil, err
	}
	out := make(map[string]string, int(count))
	ks, vs := unsafe.Slice(keys, int(count)), unsafe.Slice(values, int(count))
	for i, k := range ks {
		if k == nil || vs[i] == nil {
			return nil, fmt.Errorf("ort: nil device metadata key or value")
		}
		out[C.GoString(k)] = C.GoString(vs[i])
	}
	return out, nil
}

func snapshotEPDevice(ptr *C.OrtEpDevice) (*EPDevice, error) {
	d := &EPDevice{handle: ptr, generation: environmentGeneration}
	var name, vendor *C.char
	if err := checkStatus(C.ort_EpDevice_EpName(ptr, &name)); err != nil {
		return nil, err
	}
	if name == nil {
		return nil, fmt.Errorf("ort: nil EP name")
	}
	if err := checkStatus(C.ort_EpDevice_EpVendor(ptr, &vendor)); err != nil {
		return nil, err
	}
	d.epName, d.epVendor = C.GoString(name), C.GoString(vendor)
	var metadata, options *C.OrtKeyValuePairs
	if err := checkStatus(C.ort_EpDevice_EpMetadata(ptr, &metadata)); err != nil {
		return nil, err
	}
	if err := checkStatus(C.ort_EpDevice_EpOptions(ptr, &options)); err != nil {
		return nil, err
	}
	var err error
	d.epMetadata, err = snapshotEPMap(metadata)
	if err != nil {
		return nil, err
	}
	d.epOptions, err = snapshotEPMap(options)
	if err != nil {
		return nil, err
	}
	var hardware *C.OrtHardwareDevice
	if err := checkStatus(C.ort_EpDevice_Device(ptr, &hardware)); err != nil {
		return nil, err
	}
	if hardware == nil {
		return nil, fmt.Errorf("ort: nil hardware device")
	}
	var ty C.OrtHardwareDeviceType
	var vendorID, deviceID C.uint32_t
	if err := checkStatus(C.ort_HardwareDevice_Type(hardware, &ty)); err != nil {
		return nil, err
	}
	if err := checkStatus(C.ort_HardwareDevice_VendorId(hardware, &vendorID)); err != nil {
		return nil, err
	}
	if err := checkStatus(C.ort_HardwareDevice_DeviceId(hardware, &deviceID)); err != nil {
		return nil, err
	}
	if err := checkStatus(C.ort_HardwareDevice_Vendor(hardware, &vendor)); err != nil {
		return nil, err
	}
	if err := checkStatus(C.ort_HardwareDevice_Metadata(hardware, &metadata)); err != nil {
		return nil, err
	}
	d.device = HardwareDevice{deviceType: HardwareDeviceType(ty), vendorID: uint32(vendorID), deviceID: uint32(deviceID), vendor: C.GoString(vendor)}
	d.device.metadata, err = snapshotEPMap(metadata)
	if err != nil {
		return nil, err
	}
	return d, nil
}

// AppendExecutionProviderV2 selects discovered devices from one provider. Requires API 22.
// A native attempt protects the options from plugin unload until Close, even on failure.
func (o *SessionOptions) AppendExecutionProviderV2(devices []*EPDevice, options map[string]string) error {
	if err := checkPluginInit("SessionOptionsAppendExecutionProvider_V2"); err != nil {
		return err
	}
	if err := o.lockUsable("append execution provider V2"); err != nil {
		return err
	}
	defer o.unlock()
	mu.Lock()
	defer mu.Unlock()
	if err := checkInit(); err != nil {
		return err
	}
	if err := requirePluginAPI("SessionOptionsAppendExecutionProvider_V2"); err != nil {
		return err
	}
	if len(devices) == 0 {
		return fmt.Errorf("ort: append execution provider V2: empty device list")
	}
	if err := validateStringMap(options, "execution provider option"); err != nil {
		return err
	}
	pointers := make([]*C.OrtEpDevice, len(devices))
	seen := make(map[*C.OrtEpDevice]bool, len(devices))
	provider := ""
	for i, d := range devices {
		if d == nil || d.handle == nil || d.generation != environmentGeneration {
			return fmt.Errorf("ort: append execution provider V2: nil, zero or stale device")
		}
		if seen[d.handle] {
			return fmt.Errorf("ort: append execution provider V2: duplicate device")
		}
		seen[d.handle] = true
		if i > 0 && d.epName != provider {
			return fmt.Errorf("ort: append execution provider V2: mixed providers")
		}
		provider = d.epName
		pointers[i] = d.handle
	}
	keys, values := sortedKV(options)
	defer freeStringArrays(keys, values)
	var kp, vp **C.char
	if len(keys) > 0 {
		kp, vp = &keys[0], &values[0]
	}
	o.protectLocked()
	err := checkStatus(C.ort_SessionOptionsAppendExecutionProvider_V2(o.handle, env, &pointers[0], C.size_t(len(pointers)), kp, vp, C.size_t(len(keys))))
	if err == nil {
		o.successfulV2 = true
	}
	return wrapErr("append execution provider V2", err)
}

// SetEPSelectionPolicy chooses an automatic device policy. Requires API 22.
// A native attempt protects the options from plugin unload until Close, even on failure.
func (o *SessionOptions) SetEPSelectionPolicy(policy ExecutionProviderDevicePolicy) error {
	if err := checkPluginInit("SessionOptionsSetEpSelectionPolicy"); err != nil {
		return err
	}
	if err := o.lockUsable("set EP selection policy"); err != nil {
		return err
	}
	defer o.unlock()
	mu.Lock()
	defer mu.Unlock()
	if err := checkInit(); err != nil {
		return err
	}
	if err := requirePluginAPI("SessionOptionsSetEpSelectionPolicy"); err != nil {
		return err
	}
	if policy < ExecutionProviderDevicePolicyDefault || policy > ExecutionProviderDevicePolicyMinOverallPower {
		return fmt.Errorf("ort: invalid EP selection policy %d", policy)
	}
	if o.resources.source != nil {
		return policyBufferError()
	}
	o.protectLocked()
	o.policyAttempted = true
	return wrapErr("set EP selection policy", checkStatus(C.ort_SessionOptionsSetEpSelectionPolicy(o.handle, C.OrtExecutionProviderDevicePolicy(policy))))
}

// protectLocked requires both the options mutex and package mu.
func (o *SessionOptions) protectLocked() {
	if !o.protected {
		o.protected = true
		protectedOptionsCount++
	}
}

func checkPluginInit(operation string) error {
	mu.Lock()
	defer mu.Unlock()
	if err := checkInit(); err != nil {
		return err
	}
	return requirePluginAPI(operation)
}
