import onnx
import tml_pb2
import sys

from google.protobuf import text_format

proto = tml_pb2.TraditionalMLData()

# Read the existing address book.
f = open(sys.argv[1], "rb")
proto.ParseFromString(f.read())
f.close()
print(proto.name)