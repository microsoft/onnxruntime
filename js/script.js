var supportedOperatingSystems = new Map([
    ['linux', 'linux'],
    ['mac', 'macos'],
    ['win', 'windows'],
]);
var supportedOperatingSystemsNew = [
    {key: 'linux', value: 'linux'},
    {key: 'mac', value: 'macos'},
    {key: 'win', value: 'windows'}
]

var opts = {
    os: getAnchorSelectedOS() || getDefaultSelectedOS(),
    architecture: 'X64',
    language: 'Python(3.5-3.7)',
    hardwareAcceleration: 'DefaultCPU',
};
var ot_opts = {
    // os: getAnchorSelectedOS() || getDefaultSelectedOS(),
    ot_os: 'ot_linux',
    ot_architecture: 'ot_X64',
    ot_language: 'ot_PyTorch',
    ot_hardwareAcceleration: 'ot_CUDA',
};

var os = $(".os > .r-option");

var architecture = $(".architecture > .r-option");
var language = $(".language > .r-option");
var hardwareAcceleration = $(".hardwareAcceleration > .r-option");

var ot_os = $(".ot_os > .r-option");

var ot_architecture = $(".ot_architecture > .r-option");
var ot_language = $(".ot_language > .r-option");
var ot_hardwareAcceleration = $(".ot_hardwareAcceleration > .r-option");

function checkKeyPress(event) {
    var keycode = (event.keyCode ? event.keyCode : event.which);
    if (keycode == '13' || keycode == '32' || (keycode >= '37' && keycode <= '40')) {
        return true;
    } else {
        return false;
    }
}


os.on("click", function () {
    selectedOption(os, this, "os");
    
});
os.on("keypress keyup", function (event) {
    if (checkKeyPress(event)) {
        selectedOption(os, this, "os");
    }
});
ot_os.on("click", function () {
    ot_selectedOption(ot_os, this, "ot_os");
    
});
ot_os.on("keypress keyup", function (event) {
    if (checkKeyPress(event)) {
        ot_selectedOption(ot_os, this, "ot_os");
    }
});
architecture.on("click", function () {
    selectedOption(architecture, this, "architecture");
});
architecture.on("keypress keyup", function (event) {
    if (checkKeyPress(event)) {
        selectedOption(architecture, this, "architecture");
    }
});
ot_architecture.on("click", function () {
    ot_selectedOption(ot_architecture, this, "ot_architecture");
});
ot_architecture.on("keypress keyup", function (event) {
    if (checkKeyPress(event)) {
        ot_selectedOption(ot_architecture, this, "ot_architecture");
    }
});
language.on("click", function () {
    selectedOption(language, this, "language");
});
language.on("keypress keyup", function (event) {
    if (checkKeyPress(event)) {
        selectedOption(language, this, "language");
    }
});
ot_language.on("click", function () {
    ot_selectedOption(ot_language, this, "ot_language");
});
ot_language.on("keypress keyup", function (event) {
    if (checkKeyPress(event)) {
        ot_selectedOption(ot_language, this, "ot_language");
    }
});
hardwareAcceleration.on("click", function () {
    selectedOption(hardwareAcceleration, this, "hardwareAcceleration");
});
hardwareAcceleration.on("keypress keyup", function (event) {
    if (checkKeyPress(event)) {
        selectedOption(hardwareAcceleration, this, "hardwareAcceleration");
    }
});
ot_hardwareAcceleration.on("click", function () {
    ot_selectedOption(ot_hardwareAcceleration, this, "ot_hardwareAcceleration");
});
ot_hardwareAcceleration.on("keypress keyup", function (event) {
    if (checkKeyPress(event)) {
        ot_selectedOption(ot_hardwareAcceleration, this, "ot_hardwareAcceleration");
    }
});
// Pre-select user's operating system
$(document).ready(function () {
    var userOsOption = document.getElementById(opts.os);
    var ot_userOsOption = document.getElementById(ot_opts.ot_os);

    if (userOsOption) {
        selectedOption(os, userOsOption, "os");
    }
    if (ot_userOsOption) {
        ot_selectedOption(ot_os, ot_userOsOption, "ot_os");
    }
});


// determine os (mac, linux, windows) based on user's platform
function getDefaultSelectedOS() {
    var platform = navigator.platform.toLowerCase();
    for (var idx = 0; idx < supportedOperatingSystemsNew.length; idx++ ) {
        if (platform.indexOf(supportedOperatingSystemsNew[idx].key) !== -1) {
            return supportedOperatingSystemsNew[idx].value;
        }
    }
    // Just return something if user platform is not in our supported map
    return supportedOperatingSystemsNew[0].value;
}

// determine os based on location hash
function getAnchorSelectedOS() {
    var anchor = location.hash; 
    var ANCHOR_REGEX = /^#[^ ]+$/;
    // Look for anchor in the href
    if (!ANCHOR_REGEX.test(anchor)) {
        return false;
    }
    // Look for anchor with OS in the first portion
    var testOS = anchor.slice(1).split("-")[0];
    for (var idx = 0; idx < supportedOperatingSystemsNew.length; idx++ ) {
        if (testOS.indexOf(supportedOperatingSystemsNew[idx].key) !== -1) {
            return supportedOperatingSystemsNew[idx].value;
        }
    }
    return false;
}

function selectedOption(option, selection, category) {
    $(option).removeClass("selected");
    $(selection).addClass("selected");
    opts[category] = selection.id;
    commandMessage(buildMatcher());
}

function ot_selectedOption(option, selection, category) {
    $(option).removeClass("selected");
    $(selection).addClass("selected");
    ot_opts[category] = selection.id;
    ot_commandMessage(ot_buildMatcher());
}

function display(selection, id, category) {
    var container = document.getElementById(id);
    // Check if there's a container to display the selection
    if (container === null) {
        return;
    }
    var elements = container.getElementsByClassName(category);
    for (var i = 0; i < elements.length; i++) {
        if (elements[i].classList.contains(selection)) {
            $(elements[i]).addClass("selected");
        } else {
            $(elements[i]).removeClass("selected");
        }
    }
}

function buildMatcher() {
    return (
        opts.os +
        "," +
        opts.language +
        "," +
        opts.architecture +
        "," +
        opts.hardwareAcceleration 
    );
}

function ot_buildMatcher() {
    return (
        ot_opts.ot_os +
        "," +
        ot_opts.ot_language +
        "," +
        ot_opts.ot_architecture +
        "," +
        ot_opts.ot_hardwareAcceleration 
    );
}

function ot_commandMessage(key) {
    //console.log('key- '+key);
     var ot_object = {
        "ot_linux,ot_PyTorch,ot_X64,ot_CUDA":
            "Follow sample notebook from <a href='https://github.com/microsoft/onnxruntime-training-examples' target='_blank'>here</a>",

        "ot_linux,ot_TensorFlow,ot_X64,ot_CUDA":
            "Coming Soon",
     };
     if (!ot_object.hasOwnProperty(key)) {
        $("#ot_command span").html(
            "Coming Soon"
        );
    } else {
        $("#ot_command span").html(ot_object[key]);
    }
}

function commandMessage(key) {
   // console.log('key- '.key);
    var object = {
       
        "windows,C,X64,CUDA":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

        "windows,C++,X64,CUDA":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

        "windows,C#,X64,CUDA":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

        "windows,Python(3.5-3.7),X64,CUDA":
            "pip install onnxruntime-gpu",

        "linux,C,X64,CUDA":
            "Download .tgz file from&nbsp;<a href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

        "linux,C++,X64,CUDA":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

        "linux,C#,X64,CUDA":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

        "linux,Python(3.5-3.7),X64,CUDA":
            "pip install onnxruntime-gpu",

        "windows,C,ARM32,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "windows,C++,ARM32,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "windows,C#,ARM32,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),ARM32,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "mac,C,ARM32,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,ARM32,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM32,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),ARM32,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,C,ARM32,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,C++,ARM32,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,C#,ARM32,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,Python(3.5-3.7),ARM32,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "windows,C,ARM64,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "windows,C++,ARM64,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "windows,C#,ARM64,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),ARM64,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "linux,C,ARM32,DefaultCPU":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

        "linux,C++,ARM32,DefaultCPU":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

        "linux,Python(3.5-3.7),ARM32,DefaultCPU":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

        "windows,C,X64,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

        "windows,C,X86,DefaultCPU":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

        "windows,C++,X64,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",
            
        "windows,C++,X86,DefaultCPU":
        	"Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",
        
        "windows,C#,X64,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",
            
        "windows,C#,X86,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",


        "linux,C,X64,DefaultCPU":
            "Download .tgz file from&nbsp;<a href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

        "linux,C++,X64,DefaultCPU":
            "Download .tgz file from&nbsp;<a href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

        "linux,C#,X64,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

        "mac,C,X64,DefaultCPU":
            "Download .tgz file from&nbsp;<a href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

        "mac,C++,X64,DefaultCPU":
            "Download .tgz file from&nbsp;<a href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

        "mac,C#,X64,DefaultCPU":
            "Download .tgz file from&nbsp;<a href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

        "windows,Python(3.5-3.7),X64,DefaultCPU":
            "pip install onnxruntime",

        "mac,Python(3.5-3.7),X64,DefaultCPU":
            "pip install onnxruntime",

        "linux,Python(3.5-3.7),X64,DefaultCPU":
            "pip install onnxruntime",

        "linux,Python(3.5-3.7),ARM64,DefaultCPU":
            "pip install onnxruntime",

        "windows,C,X64,DNNL":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

        "windows,C++,X64,DNNL": 
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

        "windows,C#,X64,DNNL":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),X64,DNNL":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

        "linux,C,X64,DNNL": 
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

        "linux,C++,X64,DNNL":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

        "linux,C#,X64,DNNL":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

        "linux,Python(3.5-3.7),X64,DNNL": 
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

        "windows,C,X64,MKL-ML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",

        "windows,C++,X64,MKL-ML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",

        "windows,C#,X64,MKL-ML": "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",

        "linux,C,X64,MKL-ML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",

        "linux,C++,X64,MKL-ML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",

        "linux,C#,X64,MKL-ML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",

        "mac,C,X64,MKL-ML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",

        "mac,C++,X64,MKL-ML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",
        
        "mac,C#,X64,MKL-ML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML' target='_blank'>Microsoft.ML.OnnxRuntime.MKLML</a>",

        "linux,C,X64,nGraph":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

        "linux,C++,X64,nGraph":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

        "linux,C#,X64,nGraph":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

        "linux,Python(3.5-3.7),X64,nGraph":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

        "windows,C,X64,nGraph":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

        "windows,C++,X64,nGraph":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

        "windows,C#,X64,nGraph":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),X64,nGraph":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

        "windows,C,X64,NUPHAR":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

        "windows,C++,X64,NUPHAR":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

        "windows,C#,X64,NUPHAR":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),X64,NUPHAR":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

        "linux,C,X64,NUPHAR":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

        "linux,C++,X64,NUPHAR":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

        "linux,C#,X64,NUPHAR":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

        "linux,Python(3.5-3.7),X64,NUPHAR":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

        "linux,C,X64,OpenVINO":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

        "linux,C++,X64,OpenVINO":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

        "linux,C#,X64,OpenVINO":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

        "linux,Python(3.5-3.7),X64,OpenVINO":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

        "windows,C,X64,OpenVINO":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

        "windows,C++,X64,OpenVINO":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

        "windows,C#,X64,OpenVINO":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),X64,OpenVINO":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

        "windows,C,X64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "windows,C++,X64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "windows,C#,X64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),X64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "linux,C,X64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "linux,C++,X64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "linux,C#,X64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "linux,Python(3.5-3.7),X64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "linux,C,ARM64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "linux,C++,ARM64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "linux,C#,ARM64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "linux,Python(3.5-3.7),ARM64,TensorRT":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

        "mac,C,X64,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,X86,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,ARM32,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,ARM32,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X64,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X86,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM32,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),ARM32,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X64,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X86,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,ARM32,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,ARM32,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),X64,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),X86,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM32,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),ARM32,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,X64,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,X86,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,ARM32,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,ARM32,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X64,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X86,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM32,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),ARM32,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X64,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X86,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,ARM64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,ARM64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),X64,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),X86,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),ARM64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,X64,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,X86,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,ARM32,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,ARM32,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X64,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X86,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM32,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),ARM32,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X64,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X86,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,ARM64,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),X64,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),X86,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,ARM64,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM64,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,C,ARM64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,C++,ARM64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),ARM64,nGraph":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,ARM64,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,C#,ARM64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,Python(3.5-3.7),ARM64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,ARM64,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM64,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "windows,C,X64,DirectML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

        "windows,C++,X64,DirectML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

        "mac,Python(3.5-3.7),ARM64,CUDA":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,ARM64,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "windows,C#,X64,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),X64,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "mac,C++,ARM64,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM64,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,X64,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,X86,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),ARM64,TensorRT":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C,X64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X64,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X86,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C++,X64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X64,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,X86,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,Python(3.5-3.7),X64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,C,ARM64,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,Python(3.5-3.7),X64,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,Python(3.5-3.7),X86,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,C++,ARM64,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "mac,C#,ARM64,OpenVINO":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,Python(3.5-3.7),ARM64,OpenVINO":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C,X86,NUPHAR":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C++,X86,NUPHAR":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,C#,X86,NUPHAR":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,Python(3.5-3.7),X86,NUPHAR":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C,X86,DefaultCPU":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C++,X86,DefaultCPU":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,C#,X86,DefaultCPU":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,Python(3.5-3.7),X86,DefaultCPU":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C,X86,MKL-ML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C++,X86,MKL-ML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C#,X86,MKL-ML":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,Python(3.5-3.7),X86,MKL-ML":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,C,X86,DNNL":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C++,X86,DNNL":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,C#,X86,DNNL":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,Python(3.5-3.7),X86,DNNL":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "linux,C,X64,DirectML":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "linux,C++,X64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "linux,C#,X64,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "linux,Python(3.5-3.7),X64,DirectML":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "windows,C,X86,DirectML":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",
        
        "windows,C++,X86,DirectML":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

        "windows,C#,X86,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "windows,Python(3.5-3.7),X86,DirectML":
            "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

        "mac,C,X86,DirectML":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "mac,C++,X86,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,C#,X86,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "mac,Python(3.5-3.7),X86,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",

        "linux,C,X86,DirectML":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
        
        "linux,C++,X86,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "linux,C#,X86,DirectML":
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
            
        "linux,Python(3.5-3.7),X86,DirectML":
        	"This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>.",
		
		"linux,Java,X64,DefaultCPU":
			"Follow&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and&nbsp;<a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",
			
		"linux,Java,X64,CUDA":
			"Follow&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and&nbsp;<a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",
			
		"mac,Java,X64,DefaultCPU":
			"Follow&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and&nbsp;<a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",


        "windows,WinRT,X86,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

        "windows,WinRT,X64,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

        "windows,WinRT,ARM64,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

        "windows,WinRT,ARM32,DefaultCPU":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

        "windows,WinRT,X86,DirectML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

        "windows,WinRT,X64,DirectML":
            "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

        "windows,Java,X64,DefaultCPU":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "windows,Java,X64,CUDA":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "windows,Java,X64,TensorRT":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "windows,Java,X64,DNNL":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "windows,Java,X64,MKL-ML":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "windows,Java,X64,nGraph":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "windows,Java,X64,NUPHAR":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "windows,Java,X64,OpenVINO":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "linux,Java,X64,TensorRT":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "linux,Java,X64,DNNL":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "linux,Java,X64,MKL-ML":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "linux,Java,X64,nGraph":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "linux,Java,X64,NUPHAR":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

        "linux,Java,X64,OpenVINO":
            "Follow <a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    };

    if (!object.hasOwnProperty(key)) {
        $("#command span").html(
            "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://github.com/microsoft/onnxruntime/blob/master/BUILD.md' target='_blank'>build from source</a>."
        );
    } else {
        $("#command span").html(object[key]);
    }
}


//Accesibility Get started tabel
var KEYCODE = {
    DOWN: 40,
    LEFT: 37,
    RIGHT: 39,
    SPACE: 32,
    UP: 38
}
window.addEventListener('load', function () {
    var radiobuttons = document.querySelectorAll('[role=option]');
    for (var i = 0; i < radiobuttons.length; i++) {
        var rb = radiobuttons[i];
        rb.addEventListener('click', clickRadioGroup);
        rb.addEventListener('keydown', keyDownRadioGroup);
        rb.addEventListener('focus', focusRadioButton);
        rb.addEventListener('blur', blurRadioButton);
    }
});

function firstRadioButton(node) {
    var first = node.parentNode.firstChild;
    while (first) {
        if (first.nodeType === Node.ELEMENT_NODE) {
            if (first.getAttribute("role") === 'option') return first;
        }
        first = first.nextSibling;
    }
    return null;
}

function lastRadioButton(node) {
    var last = node.parentNode.lastChild;
    while (last) {
        if (last.nodeType === Node.ELEMENT_NODE) {
            if (last.getAttribute("role") === 'option') return last;
        }
        last = last.previousSibling;
    }
    return last;
}

function nextRadioButton(node) {
    var next = node.nextSibling;
    while (next) {
        if (next.nodeType === Node.ELEMENT_NODE) {
            if (next.getAttribute("role") === 'option') return next;
        }
        next = next.nextSibling;
    }
    return null;
}

function previousRadioButton(node) {
    var prev = node.previousSibling;
    while (prev) {
        if (prev.nodeType === Node.ELEMENT_NODE) {
            if (prev.getAttribute("role") === 'option') return prev;
        }
        prev = prev.previousSibling;
    }
    return null;
}

function getImage(node) {
    var child = node.firstChild;
    while (child) {
        if (child.nodeType === Node.ELEMENT_NODE) {
            if (child.tagName === 'IMG') return child;
        }
        child = child.nextSibling;
    }
    return null;
}

function setRadioButton(node, state) {
    
    var image = getImage(node);
    if (state == 'true') {
        node.setAttribute('aria-selected', 'true');
        // $(node).trigger()
        node.tabIndex = 0;
        node.focus();
    }
    else {
        node.setAttribute('aria-selected', 'false');
        node.tabIndex = -1;
    }
}

function clickRadioGroup(event) {

    var type = event.type;
    if (type === 'click') {
        var node = event.currentTarget;
        var radioButton = firstRadioButton(node);
        while (radioButton) {
            setRadioButton(radioButton, "false");
            radioButton = nextRadioButton(radioButton);
        }
        setRadioButton(node, "true");
        event.preventDefault();
        event.stopPropagation();
    }
}

function keyDownRadioGroup(event) {
    
    var type = event.type;
    var next = false;
    if (type === "keydown") {
        var node = event.currentTarget;
        switch (event.keyCode) {
            case KEYCODE.DOWN:
            case KEYCODE.RIGHT:
                var next = nextRadioButton(node);
                if (!next) next = firstRadioButton(node); //if node is the last node, node cycles to first.
                break;
            case KEYCODE.UP:
            case KEYCODE.LEFT:
                next = previousRadioButton(node);
                if (!next) next = lastRadioButton(node); //if node is the last node, node cycles to first.
                break;
            case KEYCODE.SPACE:
                next = node;
                break;
        }
        if (next) {
            var radioButton = firstRadioButton(node);
            while (radioButton) {
                setRadioButton(radioButton, "false");
                radioButton = nextRadioButton(radioButton);
            }
            setRadioButton(next, "true");
            event.preventDefault();
            event.stopPropagation();
        }
    }
}

function focusRadioButton(event) {
    event.currentTarget.className += ' focus';
    document.getElementById("command").innerHTML;
}

function blurRadioButton(event) {
    event.currentTarget.className = event.currentTarget.className.replace(' focus', '');
}



(function($) { 
    "use strict"; 
  
  // Carousel Extension
    // ===============================
    
        $('.carousel').each(function (index) {
        
          // This function positions a highlight box around the tabs in the tablist to use in focus styling
          
          function setTablistHighlightBox() {
  
            var $tab
                , offset
                , height
                , width
                , highlightBox = {}
  
              highlightBox.top     = 0
            highlightBox.left    = 32000
            highlightBox.height  = 0
            highlightBox.width   = 0
  
            for (var i = 0; i < $tabs.length; i++) {
              $tab = $tabs[i]
              offset = $($tab).offset()
              height = $($tab).height()
              width  = $($tab).width()
            
              if (highlightBox.top < offset.top) { 
                highlightBox.top    = Math.round(offset.top)
              }
  
              if (highlightBox.height < height) { 
                highlightBox.height = Math.round(height)
              }
              
              if (highlightBox.left > offset.left) {
                highlightBox.left = Math.round(offset.left)
              }
            
              var w = (offset.left - highlightBox.left) + Math.round(width)
            
              if (highlightBox.width < w) {
                highlightBox.width = w 
              }
                
            } // end for
  
            $tablistHighlight.style.top    = (highlightBox.top    - 2)  + 'px'
            $tablistHighlight.style.left   = (highlightBox.left   - 2)  + 'px'
            $tablistHighlight.style.height = (highlightBox.height + 7)  + 'px'
            $tablistHighlight.style.width  = (highlightBox.width  + 8)  + 'px'
          
          } // end function
        
          var $this = $(this)
            , $prev        = $this.find('[data-slide="prev"]')
            , $next        = $this.find('[data-slide="next"]')
            , $tablist    = $this.find('.carousel-indicators')
            , $tabs       = $this.find('.carousel-indicators li')
            , $tabpanels  = $this.find('.carousel-item')
            , $tabpanel
            , $tablistHighlight
            , $pauseCarousel
            , $complementaryLandmark
            , $tab
            , $is_paused = false
            , offset
            , height
            , width
            , i
            , id_title  = 'id_title'
            , id_desc   = 'id_desc'
  
  
          $tablist.attr('role', 'tablist')
          
          $tabs.focus(function() {
            $this.carousel('pause')
            $is_paused = true
            $pauseCarousel.innerHTML = "<span class='fa fa-pause' aria-hidden='true'></span>"
            $(this).parent().addClass('active');
  //          $(this).addClass('focus')
            setTablistHighlightBox()
            $($tablistHighlight).addClass('focus')
            $(this).parents('.carousel').addClass('contrast')
          })
  
          $tabs.blur(function(event) {
            $(this).parent().removeClass('active');
  //          $(this).removeClass('focus')
            $($tablistHighlight).removeClass('focus')
            $(this).parents('.carousel').removeClass('contrast')
          })
  
          
          for (i = 0; i < $tabpanels.length; i++) {
            $tabpanel = $tabpanels[i]
            $tabpanel.setAttribute('role', 'tabpanel')
            $tabpanel.setAttribute('id', 'tabpanel-' + index + '-' + i)
            $tabpanel.setAttribute('aria-labelledby', 'tab-' + index + '-' + i)
          }
  
          if (typeof $this.attr('role') !== 'string') {
            // $this.attr('role', 'complementary');
            $this.attr('aria-labelledby', id_title);
            $this.attr('aria-describedby', id_desc);
            $this.prepend('<p  id="' + id_desc   + '" class="sr-only">A carousel is a rotating set of images, rotation stops on keyboard focus on carousel tab controls or hovering the mouse pointer over images.  Use the tabs or the previous and next buttons to change the displayed slide.</p>')
            $this.prepend('<h2 id="' + id_title  + '" class="sr-only">Carousel content with ' + $tabpanels.length + ' slides.</h2>')
          }  
  
                  
          for (i = 0; i < $tabs.length; i++) {
            $tab = $tabs[i]
            
            $tab.setAttribute('role', 'tab')
            $tab.setAttribute('id', 'tab-' + index + '-' + i)
            $tab.setAttribute('aria-controls', 'tabpanel-' + index + '-' + i)
            
            var tpId = '#tabpanel-' + index + '-' + i
            var caption = $this.find(tpId).find('h1').text()
            
            if ((typeof caption !== 'string') || (caption.length === 0)) caption = $this.find(tpId).text()
            if ((typeof caption !== 'string') || (caption.length === 0)) caption = $this.find(tpId).find('h3').text()
            if ((typeof caption !== 'string') || (caption.length === 0)) caption = $this.find(tpId).find('h4').text()
            if ((typeof caption !== 'string') || (caption.length === 0)) caption = $this.find(tpId).find('h5').text()
            if ((typeof caption !== 'string') || (caption.length === 0)) caption = $this.find(tpId).find('h6').text()
            if ((typeof caption !== 'string') || (caption.length === 0)) caption = "no title";
            
  //          console.log("CAPTION: " + caption )
            
            var tabName = document.createElement('span')
            tabName.setAttribute('class', 'sr-only')
            tabName.innerHTML='Slide ' + (i+1)
            if (caption) tabName.innerHTML += ": " +  caption          
            $tab.appendChild(tabName)
            
           }
  
          // create div for focus styling of tablist
          $tablistHighlight = document.createElement('div')
          $tablistHighlight.className = 'carousel-tablist-highlight'
        //   document.body.appendChild($tablistHighlight)
          
          // create button for screen reader users to stop rotation of carousel
  
          // create button for screen reader users to pause carousel for virtual mode review
          $complementaryLandmark = document.createElement('aside')
          $complementaryLandmark.setAttribute('aria-label', 'Slider control')
          $(document.body).find('.append-play-buttom').append($complementaryLandmark)
          
          $pauseCarousel = document.createElement('button')
          $pauseCarousel.className = "carousel-pause-button"
          $pauseCarousel.innerHTML = "<span class='fa fa-pause' aria-hidden='true'></span>"
          $pauseCarousel.setAttribute('title', "Pause")
          $($complementaryLandmark).append($pauseCarousel)
          
          $($pauseCarousel).click(function() {
            if ($is_paused) {
              $pauseCarousel.innerHTML = "<span class='fa fa-pause' aria-hidden='true'></span>"
              $this.carousel('cycle')
              $is_paused = false
            }
            else {
              $pauseCarousel.setAttribute('title', "Play")
              $pauseCarousel.innerHTML = "<span class='fa fa-play' aria-hidden='true'></span>"
              $this.carousel('pause')
              $is_paused = true
            }  
          })
          $($pauseCarousel).focus(function() {
            $(this).addClass('focus')
          })
          
          $($pauseCarousel).blur(function() {
            $(this).removeClass('focus')
          })
          
          setTablistHighlightBox()
  
          $( window ).resize(function() {
            setTablistHighlightBox()
          })
          
          // Add space bar behavior to prev and next buttons for SR compatibility
          $prev.attr('aria-label', 'Previous Slide')
          $prev.keydown(function(e) {
            var k = e.which || e.keyCode
            if (/(13|32)/.test(k)) {
              e.preventDefault()
              e.stopPropagation()
              $prev.trigger('click');
            }
          });
  
          $prev.focus(function() {
            $(this).parents('.carousel').addClass('contrast')
          })        
  
          $prev.blur(function() {
            $(this).parents('.carousel').removeClass('contrast')
          })        
          
          $next.attr('aria-label', 'Next Slide')
          $next.keydown(function(e) {
            var k = e.which || e.keyCode
            if (/(13|32)/.test(k)) {
              e.preventDefault()
              e.stopPropagation()           
              $next.trigger('click');
            }
          });
  
          $next.focus(function() {
            $(this).parents('.carousel').addClass('contrast')
          })        
  
          $next.blur(function() {
            $(this).parents('.carousel').removeClass('contrast')
          })        
          
          $('.carousel-inner a').focus(function() {
            $(this).parents('.carousel').addClass('contrast')
          })        
  
           $('.carousel-inner a').blur(function() {
            $(this).parents('.carousel').removeClass('contrast')
          })        
  
          $tabs.each(function () {
              var item = $(this)
              if(item.hasClass('active')) {
                  item.attr({ 'aria-selected': 'true', 'tabindex' : '0' })
              }else{
                  item.attr({ 'aria-selected': 'false', 'tabindex' : '-1' })
              }
          })
          $("#ONNXCarousel").on('slid.bs.carousel', function(){

            $tabs.each(function () {
                var item = $(this)
                if(item.hasClass('active')) {
                    item.attr({ 'aria-selected': 'true', 'tabindex' : '0' })
                }else{
                    item.attr({ 'aria-selected': 'false', 'tabindex' : '-1' })
                }
            })
          });
        
        })
  
        var slideCarousel = $.fn.carousel.Constructor.prototype.slide;
        $.fn.carousel.Constructor.prototype.slide = function (type, next) {
          var $element = this.$element
            , $active  = $element.find('[role=tabpanel].active')
            , $next    = next || $active[type]()
            , $tab
            , $tab_count = $element.find('[role=tabpanel]').size()
            , $prev_side = $element.find('[data-slide="prev"]')
            , $next_side = $element.find('[data-slide="next"]')
            , $index      = 0
            , $prev_index = $tab_count -1
            , $next_index = 1
            , $id
          
          if ($next && $next.attr('id')) {
            $id = $next.attr('id')
            $index = $id.lastIndexOf("-")
            if ($index >= 0) $index = parseInt($id.substring($index+1), 10)
            
            $prev_index = $index - 1
            if ($prev_index < 1) $prev_index = $tab_count - 1
            
            $next_index = $index + 1
            if ($next_index >= $tab_count) $next_index = 0
          }  
                  
          $prev_side.attr('aria-label', 'Show slide ' + ($prev_index+1) + ' of ' + $tab_count)
          $next_side.attr('aria-label', 'Show slide ' + ($next_index+1) + ' of ' + $tab_count)
  
          
          slideCarousel.apply(this, arguments)
  
        $active
          .one('webkitTransitionEnd', function () {
            var $tab
            
            $tab = $element.find('li[aria-controls="' + $active.attr('id') + '"]')
            if ($tab) $tab.attr({'aria-selected':false, 'tabIndex': '-1'})
  
            $tab = $element.find('li[aria-controls="' + $next.attr('id') + '"]')
            if ($tab) $tab.attr({'aria-selected': true, 'tabIndex': '0'})
            
         })
        }
  
       var $this;
       $.fn.carousel.Constructor.prototype.keydown = function (e) {
       
       $this = $this || $(this)
       if(this instanceof Node) $this = $(this)
       
       function selectTab(index) {
         if (index >= $tabs.length) return 
         if (index < 0) return
  
         $carousel.carousel(index)
         setTimeout(function () {
              $tabs[index].focus()
              // $this.prev().focus()
         }, 150)      
       }
       
       var $carousel = $(e.target).closest('.carousel')
        , $tabs      = $carousel.find('[role=tab]')
        , k = e.which || e.keyCode
        , index
         
        if (!/(37|38|39|40)/.test(k)) return
        
        index = $tabs.index($tabs.filter('.active'))
        if (k == 37 || k == 38) {                           //  Up
        //   index--
          selectTab(index);
        }
        
        if (k == 39 || k == 40) {                          // Down
        //   index++
          selectTab(index);
        }
  
        e.preventDefault()
        e.stopPropagation()
      }
      $(document).on('keydown.carousel.data-api', 'li[role=tab]', $.fn.carousel.Constructor.prototype.keydown)
  
  
   })(jQuery);


   $(document).ready(function () {
    $(".tbl_tablist li[role='tab']").click(function () {
      $(".tbl_tablist  li[role='tab']:not(this)").attr("aria-selected", "false");
      $(this).attr("aria-selected", "true");
      var tabpanid = $(this).attr("aria-controls");
      var tabpan = $("#" + tabpanid);
      $("div[role='tabpanel']:not(tabpan)").attr("aria-hidden", "true");
      $("div[role='tabpanel']:not(tabpan)").addClass("hidden");
  
      tabpan.removeClass("hidden");
      tabpan.attr("aria-hidden", "false");
    });
  
    $(".tbl_tablist li[role='tab']").keydown(function (ev) {
      if (ev.which == 13) {
        $(this).click();
      }
    });
  
    //This adds keyboard function that pressing an arrow left or arrow right from the tabs toggel the tabs.
    $(".tbl_tablist li[role='tab']").keydown(function (ev) {
      if (ev.which == 39 || ev.which == 37) {
        var selected = $(this).attr("aria-selected");
        if (selected == "true") {
          $("li[aria-selected='false']").attr("aria-selected", "true").focus();
          $(this).attr("aria-selected", "false");
  
          var tabpanid = $("li[aria-selected='true']").attr("aria-controls");
          var tabpan = $("#" + tabpanid);
          $("div[role='tabpanel']:not(tabpan)").attr("aria-hidden", "true");
          $("div[role='tabpanel']:not(tabpan)").addClass("hidden");
  
          tabpan.attr("aria-hidden", "false");
          tabpan.removeClass("hidden");
        }
      }
    });
  });

  // Modal Extension
  // ===============================

  $('.modal-dialog').attr( {'role' : 'document'})
    var modalhide =   $.fn.modal.Constructor.prototype.hide
    $.fn.modal.Constructor.prototype.hide = function(){
       modalhide.apply(this, arguments)
       $(document).off('keydown.bs.modal')
    }

    var modalfocus =   $.fn.modal.Constructor.prototype.enforceFocus
    $.fn.modal.Constructor.prototype.enforceFocus = function(){
      var $content = this.$element.find(".modal-content")
      var focEls = $content.find(":tabbable")
      , $lastEl = $(focEls[focEls.length-1])
      , $firstEl = $(focEls[0])
      $lastEl.on('keydown.bs.modal', $.proxy(function (ev) {
        if(ev.keyCode === 9 && !(ev.shiftKey | ev.ctrlKey | ev.metaKey | ev.altKey)) { // TAB pressed
          ev.preventDefault();
          $firstEl.focus();
        }
      }, this))
      $firstEl.on('keydown.bs.modal', $.proxy(function (ev) {
          if(ev.keyCode === 9 && ev.shiftKey) { // SHIFT-TAB pressed
            ev.preventDefault();
            $lastEl.focus();
          }
      }, this))
      modalfocus.apply(this, arguments)
    }