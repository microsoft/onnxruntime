import random
import unittest


def emulate_map(mask, past):
    visible = [token <= past and value != 0 for token, value in enumerate(mask)]
    counts = [sum(visible[lane::64]) for lane in range(64)]
    stride = 32
    while stride:
        for lane in range(stride):
            counts[lane] += counts[lane + stride]
        stride //= 2
    count = counts[0]
    output = [None] * (len(mask) + 1)
    writes = [0] * len(output)

    def write(index, value):
        output[index] = value
        writes[index] += 1

    if count == 0 or count == len(mask):
        if count:
            for lane in range(64):
                for token in range(lane, len(mask), 64):
                    write(token, token)
        path = "dense_or_empty"
    else:
        offset = 0
        for base in range(0, len(mask), 64):
            flags = visible[base:base + 64]
            for lane, flag in enumerate(flags):
                if flag:
                    write(offset + sum(flags[:lane]), base + lane)
            offset += sum(flags)
        path = "fallback"
    write(len(mask), count)
    return output, writes, path


class DenseMapTests(unittest.TestCase):
    def test_exact_mapping_and_single_writers_at_boundaries(self):
        randomizer = random.Random(843)
        for length in [0, 1, 63, 64, 65, 511, 512, 513, 2048, 8192, 33024, 65536]:
            masks = [
                [1] * length, [0] * length,
                [(-1 if i % 2 else 1 << 40) for i in range(length)],
                [randomizer.choice([0, 1, -1, -(1 << 63)]) for _ in range(length)],
            ]
            if length:
                for hole in {0, length // 2, length - 1}:
                    mask = [1] * length
                    mask[hole] = 0
                    masks.append(mask)
            for mask in masks:
                for past in [length - 1, length // 2]:
                    with self.subTest(length=length, past=past):
                        output, writers, path = emulate_map(mask, past)
                        reference = [i for i, value in enumerate(mask) if i <= past and value != 0]
                        self.assertEqual(output[:len(reference)], reference)
                        self.assertEqual(output[-1], len(reference))
                        self.assertEqual(writers, [1] * len(reference) +
                                         [0] * (length - len(reference)) + [1])
                        expected_path = "dense_or_empty" if len(reference) in (0, length) else "fallback"
                        self.assertEqual(path, expected_path)


if __name__ == "__main__":
    unittest.main()
