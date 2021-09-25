deep_conflation = r"C:\Users\wangye\Work\Turing-Sep-model\DeepConflation_20x512.onnx"
deep_conflation_opt = r"C:\Users\wangye\Work\Turing-Sep-model\DeepConflation_20x512_opt.onnx"
offensive_v4 = r"C:\Users\wangye\Work\Turing-Sep-model\OffensiveV4_10x128.onnx"
offensive_v4_opt = r"C:\Users\wangye\Work\Turing-Sep-model\OffensiveV4_10x128_opt.onnx"

def parity_check(path, opt_path, input_name, mask_name, dict_size, batches, seqlens, times = 100):
    import numpy, time
    from onnxruntime import SessionOptions, InferenceSession
    sess_options = SessionOptions()
    #sess_options.intra_op_num_threads = 20
    for ep in ['CPUExecutionProvider', 'CUDAExecutionProvider']:
        session = InferenceSession(path, sess_options, providers=[ep])
        session_opt = InferenceSession(opt_path, sess_options, providers=[ep])
        for batch in batches:
            for seq_len in seqlens:
                input_ids = numpy.random.randint(low=0, high=dict_size - 1, size=(batch, seq_len), dtype = numpy.int64)
                attention_mask = numpy.random.randint(2, size=(batch, seq_len), dtype = numpy.int64)
                inputs = {input_name : input_ids, mask_name : attention_mask}
                t1 = time.time()
                for i in range(times):
                    output = session.run(None, inputs)
                t2 = time.time()
                for i in range(times):
                    output_opt = session_opt.run(None, inputs)
                t3 = time.time()
                span_1 = (t2 - t1)/times
                span_2 = (t3 - t2)/times
                print("-------------------ONNXRuntime result-------------------")
                print("model:", opt_path)
                print("device:", ep)
                print("batch:", batch, "seq_len:", seq_len)
                print("model time:", span_1, "opt model time:", span_2)
                print("speed up:", numpy.round(span_1/span_2*100 - 100), "%")
                for i in range(len(output)):
                    print("parity result for output", i, ":", numpy.allclose(output[i], output_opt[i], 1e-3, 1e-3))

from optimizer import optimize_model
from fusion_options import FusionOptions

export = False

if export:
    optimization_options = FusionOptions('bert')
    opt_model = optimize_model(deep_conflation,
                               'bert',
                               num_heads=8,
                               hidden_size=256,
                               opt_level=1,
                               optimization_options=optimization_options,
                               use_gpu=False,
                               only_onnxruntime=False)
    opt_model.save_model_to_file(deep_conflation_opt, use_external_data_format=False)
    print(deep_conflation)
    print(opt_model.get_fused_operator_statistics())
    opt_model = optimize_model(offensive_v4,
                               'bert',
                               num_heads=16,
                               hidden_size=1024,
                               opt_level=1,
                               optimization_options=optimization_options,
                               use_gpu=False,
                               only_onnxruntime=False)
    opt_model.save_model_to_file(offensive_v4_opt, use_external_data_format=False)
    print(offensive_v4)
    print(opt_model.get_fused_operator_statistics())

parity_check(deep_conflation, deep_conflation_opt, 'input_ids', 'attention_mask', 64044, [20], [512])
parity_check(offensive_v4, offensive_v4_opt, 'input_ids', 'input_masks', 128000, [10], [128])
