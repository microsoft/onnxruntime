import onnx
import tml_pb2
import sys
import six

def _VarintDecoder(mask, result_type):
  """Return an encoder for a basic varint value (does not include tag).
  Decoded values will be bitwise-anded with the given mask before being
  returned, e.g. to limit them to 32 bits.  The returned decoder does not
  take the usual "end" parameter -- the caller is expected to do bounds checking
  after the fact (often the caller can defer such checking until later).  The
  decoder returns a (value, new_pos) pair.
  """

  def DecodeVarint(buffer, pos):
    result = 0
    shift = 0
    while 1:
      b = six.indexbytes(buffer, pos)
      result |= ((b & 0x7f) << shift)
      pos += 1
      if not (b & 0x80):
        result &= mask
        result = result_type(result)
        return (result, pos)
      shift += 7
      if shift >= 64:
        raise _DecodeError('Too many bytes when decoding varint.')
  return DecodeVarint

decoder = _VarintDecoder((1 << 32) - 1, int)

pb_file_name = sys.argv[1]
pos = 0
with open(pb_file_name,'rb') as f:
    buf = f.read()
    total_len = len(buf)
    file_index = 0
    while True:
        len, pos = decoder(buf,pos)        
        data = tml_pb2.TraditionalMLData()
        data.ParseFromString(buf[pos:pos+len])
        print(data.debug_info)
        pos+=len
        with open("%d.pb" % file_index,'wb') as f:
            #print(data.tensor)
            f.write(data.tensor.SerializeToString())
        if pos == total_len:
          break
print('end')