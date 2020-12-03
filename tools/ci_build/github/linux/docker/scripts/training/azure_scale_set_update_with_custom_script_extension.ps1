# https://docs.microsoft.com/en-us/azure/virtual-machine-scale-sets/tutorial-install-apps-powershell#create-custom-script-extension-definition
$customConfig = @{
  "fileUris" = (,"https://onnxruntimetestdata.file.core.windows.net/bert-data/azure_scale_set_custom_script_extension_map_test_data.sh");
  "commandToExecute" = "bash azure_scale_set_custom_script_extension_map_test_data.sh"
}

# Get information about the scale set
$vmss = Get-AzVmss `
          -ResourceGroupName "ONNX_TRAINING" `
          -VMScaleSetName "gpu_distributed_training_e2e2"

# Add the Custom Script Extension to install IIS and configure basic website
$vmss = Add-AzVmssExtension `
  -VirtualMachineScaleSet $vmss `
  -Name "customScript" `
  -Publisher "Microsoft.Compute" `
  -Type "CustomScriptExtension" `
  -TypeHandlerVersion 1.9 `
  -Setting $customConfig

# Update the scale set and apply the Custom Script Extension to the VM instances
Update-AzVmss `
  -ResourceGroupName "ONNX_TRAINING" `
  -Name "gpu_distributed_training_e2e2" `
  -VirtualMachineScaleSet $vmss
