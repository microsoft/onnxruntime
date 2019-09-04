import os
import sys
import os.path
import xml.etree.cElementTree as ET

dir = sys.argv[1]
root = ET.Element('root')
for test_name in os.listdir(dir):
    configs = ['retail']
    if test_name not in ['coreml_AgeNet_ImageNet', 'coreml_Inceptionv3_ImageNet', 'coreml_Resnet50_ImageNet', 'coreml_SqueezeNet_ImageNet', 'coreml_VGG16_ImageNet']:
        configs.append('debug')
    for config in configs:
        TestJob = ET.SubElement(root, 'TestJob', attrib= {'Name':'LotusModelTest.%s.%s' % (test_name,config),'Type':'SingleBox','Tags':'Suite=Suite0','OwnerAliases':'chasun','TimeoutMins':'60'})
        ET.SubElement(TestJob,'Execution',attrib= {'Type':'MsTest','Path':'[WorkingDirectory]\\%s\\onnx_test_runner_vstest.dll' % config})
ET.ElementTree(root).write('winml.xml',encoding='utf-8',xml_declaration=True)