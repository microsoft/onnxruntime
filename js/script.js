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
    language: 'Python(3.6-3.9)',
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

var supported = true;
var default_selection = false;

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
    default_selection = true;
    if (userOsOption) {
        selectedOption(os, userOsOption, "os");
        
    }
    if (ot_userOsOption) {
        ot_selectedOption(ot_os, ot_userOsOption, "ot_os");
    }
});


// determine os (mac, linux, windows) based on user's platform
function getDefaultSelectedOS() {
    default_selection = true;
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

function checkValidity(){
    var current_os = opts['os'];
    var current_lang = opts['language'];
    var current_arch = opts['architecture'];
    var current_hw = opts['hardwareAcceleration'];
    
    // console.log("current: "+current_os);
    // console.log("current: "+current_arch);
    // console.log("current: "+current_lang);
    // console.log("current: "+current_hw);

    var valid = Object.getOwnPropertyNames(validCombos);
  
    //os section
    for(var i =0; i<os.length; i++){
        var found=false;
        for(var k=0; k<valid.length;k++){
            if(valid[k].indexOf(os[i].id)!=-1 && valid[k].indexOf(current_arch)!=-1 && valid[k].indexOf(current_lang)!=-1 && valid[k].indexOf(current_hw)!=-1){
                found=true;
                break;       
            }
        }
     
        if(found==false && os[i].id!=current_os){
            $(os[i]).addClass("gray"); 
        }
    }

        //language section
        for(var i =0; i<language.length; i++){
            var found=false;
            for(var k=0; k<valid.length;k++){
                if(valid[k].indexOf(current_os)!=-1 && valid[k].indexOf(current_arch)!=-1 && valid[k].indexOf(language[i].id)!=-1 && valid[k].indexOf(current_hw)!=-1){
                    found=true;
                    break;       
                }
            }
           
            if(found==false && language[i].id!=current_lang){
                $(language[i]).addClass("gray"); 
            }
        }

       //architecture section
       for(var i =0; i<architecture.length; i++){
        var found=false;
        for(var k=0; k<valid.length;k++){
            if(valid[k].indexOf(current_os)!=-1 && valid[k].indexOf(architecture[i].id)!=-1 && valid[k].indexOf(current_lang)!=-1 && valid[k].indexOf(current_hw)!=-1){
                found=true;
                break;       
            }
        }
      
        if(found==false && architecture[i].id!=current_arch){
            $(architecture[i]).addClass("gray"); 
        }
    }

          //accelerator section
          for(var i =0; i<hardwareAcceleration.length; i++){
            var found=false;
            for(var k=0; k<valid.length;k++){
                if(valid[k].indexOf(current_os)!=-1 && valid[k].indexOf(current_arch)!=-1 && valid[k].indexOf(current_lang)!=-1 && valid[k].indexOf(hardwareAcceleration[i].id)!=-1){
                    found=true;
                    break;       
                }
            }
            
            if(found==false && hardwareAcceleration[i].id!=current_hw){
                $(hardwareAcceleration[i]).addClass("gray"); 
            }
        } 
}

function resetOptions(){
  for(var i=0; i<os.length;i++){
    $(os[i]).removeClass("gray");
  }
  for(var i=0; i<language.length;i++){
    $(language[i]).removeClass("gray");
  }
  for(var i=0; i<architecture.length;i++){
    $(architecture[i]).removeClass("gray");
  }
  for(var i=0; i<hardwareAcceleration.length;i++){
    $(hardwareAcceleration[i]).removeClass("gray");
  }

  default_selection=false;
}


function selectedOption(option, selection, category) {

    // console.log(opts[category]);
    if(selection.id==opts[category] && default_selection==false){
        $(selection).removeClass("selected");
        $(selection).removeClass("unsupported");
        opts[category] = '';
    }
    else{
        $(option).removeClass("selected");
    $(option).removeClass("unsupported");
    $(selection).addClass("selected");
    opts[category] = selection.id;
    }

    resetOptions();

    var all_selected = document.getElementsByClassName('selected r-option');
    
    var isSupported = commandMessage(buildMatcher());
  
    //mark unsupported combos
    
    if (isSupported==false){
        mark_unsupported(all_selected);
    }
    else{
        for(var i = 0; i<all_selected.length; i++){
            $(all_selected[i]).removeClass("unsupported");
        }
    }

    checkValidity();
}


function mark_unsupported(selection){
    for(var i = 0; i<selection.length; i++){
       if(selection[i].id.indexOf('ot_') == -1){
            $(selection[i]).addClass("unsupported");
       }
    }
    
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

var ot_validCombos = {
    "ot_linux,ot_PyTorch,ot_X64,ot_CUDA":
    "Follow sample notebook from <a href='https://github.com/microsoft/onnxruntime-training-examples' target='_blank'>here</a>",

"ot_linux,ot_TensorFlow,ot_X64,ot_CUDA":
    "Coming Soon"
};

function ot_commandMessage(key) {
    if(ot_opts['os']=='' || ot_opts['architecture'] == '' || ot_opts['language']=='' || ot_opts['hardwareAcceleration'] == ''){
        $("#ot_command span").html(
            "Please complete your selection"
        ) 
    }
    else if (!ot_validCombos.hasOwnProperty(key)) {
        $("#ot_command span").html(
            "This combination is not supported"
        ) 
        return false;
    } else {
        $("#ot_command span").html(ot_validCombos[key]);
        return true;
    }

    // //console.log('key- '+key);
    //  var ot_object = {
    //     "ot_linux,ot_PyTorch,ot_X64,ot_CUDA":
    //         "Follow sample notebook from <a href='https://github.com/microsoft/onnxruntime-training-examples' target='_blank'>here</a>",

    //     "ot_linux,ot_TensorFlow,ot_X64,ot_CUDA":
    //         "Coming Soon",
    //  };
    //  if (!ot_object.hasOwnProperty(key)) {
    //     $("#ot_command span").html(
    //         "Coming Soon"
    //     );
    // } else {
    //     $("#ot_command span").html(ot_object[key]);
    // }
}

var validCombos = {
       
    "windows,C,X64,CUDA":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

    "windows,C++,X64,CUDA":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

    "windows,C#,X64,CUDA":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

    "windows,Python(3.6-3.9),X64,CUDA":
        "pip install onnxruntime-gpu",

"linux,Python(3.6-3.9),ARM64,CUDA":
        "For Jetpack 4.4+, follow installation instructions from <a href='https://elinux.org/Jetson_Zoo#ONNX_Runtime' target='_blank'>here</a>",
    
    "linux,C,X64,CUDA":
        "Download .tgz file from&nbsp;<a href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

    "linux,C++,X64,CUDA":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

    "linux,C#,X64,CUDA":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a>",

    "linux,Python(3.6-3.9),X64,CUDA":
        "pip install onnxruntime-gpu",

    "windows,C,ARM32,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "windows,C++,ARM32,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "windows,C#,ARM32,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "windows,Python(3.6-3.9),ARM32,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "windows,C,ARM64,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "windows,C++,ARM64,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "windows,C#,ARM64,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "windows,Python(3.6-3.9),ARM64,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "linux,C,ARM32,DefaultCPU":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

    "linux,C++,ARM32,DefaultCPU":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

    "linux,Python(3.6-3.9),ARM32,DefaultCPU":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

    "windows,C,X64,DefaultCPU":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

    "windows,C,X86,DefaultCPU":
    "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

"windows,C,ARM32,DefaultCPU":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",
    
"windows,C++,ARM32,DefaultCPU":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",
    
"windows,C#,ARM32,DefaultCPU":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",
    
"windows,C,ARM64,DefaultCPU":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",
     
"windows,C++,ARM64,DefaultCPU":
        "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",
    
"windows,C#,ARM64,DefaultCPU":
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

    "windows,Python(3.6-3.9),X64,DefaultCPU":
        "pip install onnxruntime",

    "mac,Python(3.6-3.9),X64,DefaultCPU":
        "pip install onnxruntime",

    "linux,Python(3.6-3.9),X64,DefaultCPU":
        "pip install onnxruntime",

    "linux,Python(3.6-3.9),ARM64,DefaultCPU":
        "pip install onnxruntime",

    "windows,C,X64,DNNL":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

    "windows,C++,X64,DNNL": 
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

    "windows,C#,X64,DNNL":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

    "windows,Python(3.6-3.9),X64,DNNL":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

    "linux,C,X64,DNNL": 
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

    "linux,C++,X64,DNNL":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

    "linux,C#,X64,DNNL":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

    "linux,Python(3.6-3.9),X64,DNNL": 
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

    "linux,C,X64,nGraph":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

    "linux,C++,X64,nGraph":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

    "linux,C#,X64,nGraph":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

    "linux,Python(3.6-3.9),X64,nGraph":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

    "windows,C,X64,nGraph":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

    "windows,C++,X64,nGraph":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

    "windows,C#,X64,nGraph":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

    "windows,Python(3.6-3.9),X64,nGraph":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-ngraph' target='_blank'>here</a>",

    "windows,C,X64,NUPHAR":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

    "windows,C++,X64,NUPHAR":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

    "windows,C#,X64,NUPHAR":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

    "windows,Python(3.6-3.9),X64,NUPHAR":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

    "linux,C,X64,NUPHAR":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

    "linux,C++,X64,NUPHAR":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

    "linux,C#,X64,NUPHAR":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

    "linux,Python(3.6-3.9),X64,NUPHAR":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-nuphar' target='_blank'>here</a>",

    "linux,C,X64,OpenVINO":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

    "linux,C++,X64,OpenVINO":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

    "linux,C#,X64,OpenVINO":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

    "linux,Python(3.6-3.9),X64,OpenVINO":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

    "windows,C,X64,OpenVINO":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

    "windows,C++,X64,OpenVINO":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

    "windows,C#,X64,OpenVINO":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

    "windows,Python(3.6-3.9),X64,OpenVINO":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

    "windows,C,X64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "windows,C++,X64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "windows,C#,X64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "windows,Python(3.6-3.9),X64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "linux,C,X64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "linux,C++,X64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "linux,C#,X64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "linux,Python(3.6-3.9),X64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "linux,C,ARM64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "linux,C++,ARM64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "linux,C#,ARM64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "linux,Python(3.6-3.9),ARM64,TensorRT":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-tensorrt' target='_blank'>here</a>",

    "mac,C,X86,DefaultCPU":
        "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://www.onnxruntime.ai/docs/how-to/build.html' target='_blank'>build from source</a>.",
        
    "mac,C++,X86,DefaultCPU":
        "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://www.onnxruntime.ai/docs/how-to/build.html' target='_blank'>build from source</a>.",
    
    "mac,C#,X86,DefaultCPU":
        "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://www.onnxruntime.ai/docs/how-to/build.html' target='_blank'>build from source</a>.",
        
    "mac,Python(3.6-3.9),X86,DefaultCPU":
        "This combination of resources has not yet been tested. It may be possible to&nbsp;<a href='https://www.onnxruntime.ai/docs/how-to/build.html' target='_blank'>build from source</a>.",

    "windows,C,X86,DirectML":
    "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",
    
    "windows,C++,X86,DirectML":
    "Install Nuget package&nbsp;<a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

    "windows,C#,X86,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

    "windows,Python(3.6-3.9),X86,DirectML":
        "Follow build instructions from&nbsp;<a href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",
    
    "linux,Java,X64,DefaultCPU":
        "Add a dependency on <a href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime' target='_blank'>com.microsoft.onnxruntime:onnxruntime</a> using Maven/Gradle",
        
    "linux,Java,X64,CUDA":
        "Add a dependency on <a href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu' target='_blank'>com.microsoft.onnxruntime:onnxruntime_gpu</a> using Maven/Gradle",
        
    "linux,Javascript,X64,DefaultCPU":
        "npm install onnxruntime",
    
    "mac,Java,X64,DefaultCPU":
        "Add a dependency on <a href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime' target='_blank'>com.microsoft.onnxruntime:onnxruntime</a> using Maven/Gradle",

    "mac,Javascript,X64,DefaultCPU":
        "npm install onnxruntime",

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
        "Add a dependency on <a href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime' target='_blank'>com.microsoft.onnxruntime:onnxruntime</a> using Maven/Gradle",

    "windows,Java,X64,CUDA":
        "Add a dependency on <a href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu' target='_blank'>com.microsoft.onnxruntime:onnxruntime_gpu</a> using Maven/Gradle",

    "windows,Java,X64,TensorRT":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "windows,Java,X64,DNNL":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "windows,Java,X64,MKL-ML":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "windows,Java,X64,nGraph":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "windows,Java,X64,NUPHAR":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "windows,Java,X64,OpenVINO":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "linux,Java,X64,TensorRT":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "linux,Java,X64,DNNL":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "linux,Java,X64,MKL-ML":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "linux,Java,X64,nGraph":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "linux,Java,X64,NUPHAR":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

    "linux,Java,X64,OpenVINO":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#common-build-instructions' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",
    
    "windows,Javascript,X64,DefaultCPU":
        "npm install onnxruntime",
    
    "android,C,ARM64,NNAPI":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Android-NNAPI-Execution-Provider' target='_blank'>here</a>",
    
    "android,C++,ARM64,NNAPI":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Android-NNAPI-Execution-Provider' target='_blank'>here</a>",
    
    "android,Java,ARM64,NNAPI":
        "Follow <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Android-NNAPI-Execution-Provider' target='_blank'>build</a> and <a href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",
    
    "ios,C,ARM64,DefaultCPU":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#iOS' target='_blank'>here</a>",
    
    "ios,C++,ARM64,DefaultCPU":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#iOS' target='_blank'>here</a>",
    
    "ios,C,ARM64,CoreML":
        "Follow build instructions from <a href='https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_for_Mobile_Platforms.md' target='_blank'>here</a>",
    
    "ios,C++,ARM64,CoreML":
        "Follow build instructions from <a href='https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_for_Mobile_Platforms.md' target='_blank'>here</a>",
    
    "windows,Python(3.6-3.9),X86,VitisAI":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Vitis-AI' target='_blank'>here</a>",
    
    "windows,C,X86,VitisAI":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Vitis-AI' target='_blank'>here</a>",
    
    "windows,C++,X86,VitisAI":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Vitis-AI' target='_blank'>here</a>",
    
    "linux,Python(3.6-3.9),X86,VitisAI":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Vitis-AI' target='_blank'>here</a>",
    
    "linux,C,X86,VitisAI":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Vitis-AI' target='_blank'>here</a>",
    
    "linux,C++,X86,VitisAI":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#Vitis-AI' target='_blank'>here</a>",
    
    "windows,Python(3.6-3.9),X86,MIGraphX":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#AMD-MIGraphX' target='_blank'>here</a>",
    
    "windows,C,X86,MIGraphX":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#AMD-MIGraphX' target='_blank'>here</a>",
    
    "windows,C++,X86,MIGraphX":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#AMD-MIGraphX' target='_blank'>here</a>",
    
    "linux,Python(3.6-3.9),X86,MIGraphX":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#AMD-MIGraphX' target='_blank'>here</a>",
    
    "linux,C,X86,MIGraphX":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#AMD-MIGraphX' target='_blank'>here</a>",
    
    "linux,C++,X86,MIGraphX":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#AMD-MIGraphX' target='_blank'>here</a>",
    
    "linux,Python(3.6-3.9),ARM64,ACL":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ARM-Compute-Library' target='_blank'>here</a>",
    
    "linux,C,ARM64,ACL":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ARM-Compute-Library' target='_blank'>here</a>",
    
    "linux,C++,ARM64,ACL":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ARM-Compute-Library' target='_blank'>here</a>",
    
    "linux,Python(3.6-3.9),ARM32,ACL":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ARM-Compute-Library' target='_blank'>here</a>",
    
    "linux,C,ARM32,ACL":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ARM-Compute-Library' target='_blank'>here</a>",
    
    "linux,C++,ARM32,ACL":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ARM-Compute-Library' target='_blank'>here</a>",
    
    "linux,Python(3.6-3.9),ARM64,ArmNN":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ArmNN' target='_blank'>here</a>",
    
    "linux,C,ARM64,ArmNN":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ArmNN' target='_blank'>here</a>",
    
    "linux,C++,ARM64,ArmNN":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ArmNN' target='_blank'>here</a>",
    
    "linux,Python(3.6-3.9),ARM32,ArmNN":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ArmNN' target='_blank'>here</a>",
    
    "linux,C,ARM32,ArmNN":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ArmNN' target='_blank'>here</a>",
    
    "linux,C++,ARM32,ArmNN":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#ArmNN' target='_blank'>here</a>",
    
    "linux,Python(3.6-3.9),ARM64,RockchipNPU":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#RKNPU' target='_blank'>here</a>",
    
    "linux,C,ARM64,RockchipNPU":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#RKNPU' target='_blank'>here</a>",
    
    "linux,C++,ARM64,RockchipNPU":
        "Follow build instructions from <a href='https://www.onnxruntime.ai/docs/how-to/build.html#RKNPU' target='_blank'>here</a>",
    
    "mac,C#,X64,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,C,X64,CoreML":
        "<i>Coming soon!</i>",

    "mac,C++,X64,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,Python(3.6-3.9),X64,CoreML":
        "<i>Coming soon!</i>",

    "mac,Java,X64,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,C,X86,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,C#,X86,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,C,86,CoreML":
        "<i>Coming soon!</i>",

    "mac,C++,X86,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,Python(3.6-3.9),X86,CoreML":
        "<i>Coming soon!</i>",

    "mac,Java,X86,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,C,X86,CoreML":
        "<i>Coming soon!</i>",

    "mac,C#,ARM64,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,C,ARM64,CoreML":
        "<i>Coming soon!</i>",

    "mac,C++,ARM64,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,Python(3.6-3.9),ARM64,CoreML":
        "<i>Coming soon!</i>",

    "mac,Java,ARM64,CoreML":
        "<i>Coming soon!</i>",
    
    "mac,C,ARM64,CoreML":
        "<i>Coming soon!</i>"
    
};

function commandMessage(key) {
   // console.log('key- '.key);

    if(opts['os']=='' || opts['architecture'] == '' || opts['language']=='' || opts['hardwareAcceleration'] == ''){
        $("#command span").html(
            "Please complete your selection"
        ) 
    }
    else if (!validCombos.hasOwnProperty(key)) {
        $("#command span").html(
            "This combination is not supported"
        ) 
        return false;
    } else {
        $("#command span").html(validCombos[key]);
        return true;
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


   $(document).ready(function () {
    $(".tbl_tablist li[role='tab']").click(function () {
      $(".tbl_tablist li[role='tab']:not(this)").attr("aria-selected", "false");
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

      $(function() {
        var tabs = $(".custom-tab");
      
        // For each individual tab DIV, set class and aria role attributes, and hide it
        $(tabs).find(".tab-content > div.tab-pane").attr({
          "class": "tabPanel",
          "role": "tabpanel",
          "aria-hidden": "true"
        }).hide();
      
        // Get the list of tab links
        var tabsList = tabs.find("ul:first").attr({    
          "role": "tablist"
        });
      
        // For each item in the tabs list...
        $(tabsList).find("li > a").each(
          function(a) {
            var tab = $(this);
      
            // Create a unique id using the tab link's href
            var tabId = "tab-" + tab.attr("href").slice(1);
      
            // Assign tab id, aria and tabindex attributes to the tab control, but do not remove the href
            tab.attr({
              "id": tabId,
              "role": "tab",
              "aria-selected": "false",
            //   "tabindex": "-1"
            }).parent().attr("role", "presentation");
      
            // Assign aria attribute to the relevant tab panel
            $(tabs).find(".tabPanel").eq(a).attr("aria-labelledby", tabId);
      
            // Set the click event for each tab link
            tab.click(
              function(e) {
                // Prevent default click event
                e.preventDefault();
      
                // Change state of previously selected tabList item
                $(tabsList).find("> li.active").removeClass("active").find("> a").attr({
                  "aria-selected": "false",
                //   "tabindex": "-1"
                });
      
                // Hide previously selected tabPanel
                $(tabs).find(".tabPanel:visible").attr("aria-hidden", "true").hide();
      
                // Show newly selected tabPanel
                $(tabs).find(".tabPanel").eq(tab.parent().index()).attr("aria-hidden", "false").show();
      
                // Set state of newly selected tab list item
                tab.attr({
                  "aria-selected": "true",
                  "tabindex": "0"
                }).parent().addClass("active");
                tab.focus();
              }
            );
          }
        );
      
        // Set keydown events on tabList item for navigating tabs
        $(tabsList).delegate("a", "keydown",
          function(e) {
            var tab = $(this);
            switch (e.which) {
              case 37:
                //case 38:
                if (tab.parent().prev().length != 0) {
                  tab.parent().prev().find("> a").click();
                } else {
                  $(tabsList).find("li:last > a").click();
                }
                break;
              case 39:
                //case 40:
                if (tab.parent().next().length != 0) {
                  tab.parent().next().find("> a").click();
                } else {
                  $(tabsList).find("li:first > a").click();
                }
                break;
            }
          }
        );
      
        // Show the first tabPanel
        $(tabs).find(".tabPanel:first").attr("aria-hidden", "false").show();
      
        // Set state for the first tabsList li
        $(tabsList).find("li:first").addClass("active").find(" > a").attr({
          "aria-selected": "true",
          "tabindex": "0"
        });
      });
