"""Static stabilizer-code analysis and an external decoder boundary."""

import numpy as np

from tencirpauli import PauliWord, StabilizerCode, StabilizerErrorKind


def main() -> None:
    checks = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    code = StabilizerCode.from_css(
        x_checks=np.empty((0, 3), dtype=np.uint8),
        z_checks=checks,
    )

    errors = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    syndromes = code.syndrome_many(errors)
    batch = code.classify_many(errors)
    print("syndromes:", syndromes.tolist())
    print("kinds:", batch.kinds.tolist())

    ambiguity = code.analyze_error_set(
        np.asarray([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    )
    print("same-syndrome conflict:", ambiguity.conflicting_pair)

    residual = code.analyze_corrections(
        errors=np.asarray([[0, 1, 1], [1, 0, 0]], dtype=np.uint8),
        corrections=np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.uint8),
    )
    assert residual.kinds[0] == StabilizerErrorKind.LOGICAL
    print("decoder residual kinds:", residual.kinds.tolist())

    logical_x, logical_z = code.logical_operators()
    assert logical_x[0].commutes_with(PauliWord.from_string("ZZI"))
    assert logical_z[0].commutes_with(PauliWord.from_string("ZZI"))

    five_qubit = StabilizerCode.from_generators(
        5,
        [
            PauliWord.from_string("XZZXI"),
            PauliWord.from_string("IXZZX"),
            PauliWord.from_string("XIXZZ"),
            PauliWord.from_string("ZXIXZ"),
        ],
    )
    five_x, five_z = five_qubit.logical_operators()
    assert five_x[0].symplectic_inner_product(five_z[0]) == 1
    assert five_x[0].commutes_with(PauliWord.from_string("XZZXI"))
    print("five-qubit logical pair:", five_x[0], five_z[0])


if __name__ == "__main__":
    main()
