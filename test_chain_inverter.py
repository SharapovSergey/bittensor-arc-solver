"""
Functional test for chain inversion (Phase 3 / #2).

Builds synthetic ARC tasks with known chain, verifies inverter recovers it.
"""
import arc_chain_inverter as inv


def test_self_inverse_ops():
    """flip_h(flip_h(g)) == g, etc."""
    g = [[1, 2, 3], [4, 5, 6]]
    for op_name, op_fn in [("flip_horizontal", inv.flip_horizontal),
                           ("flip_vertical",   inv.flip_vertical),
                           ("rotate_180",      inv.rotate_180),
                           ("transpose",       inv.transpose),
                           ("flip_diagonal",   inv.flip_diagonal)]:
        once = op_fn(g)
        twice = op_fn(once)
        assert inv._grids_equal(twice, g), f"{op_name} not self-inverse: {twice} != {g}"
    print("  ✅ test_self_inverse_ops")


def test_rotate_pair():
    """rotate_270(rotate_90(g)) == g"""
    g = [[1, 2, 3], [4, 5, 6]]
    once = inv.rotate_90(g)
    twice = inv.rotate_270(once)
    assert inv._grids_equal(twice, g), f"rot_270∘rot_90 != identity: {twice}"
    print("  ✅ test_rotate_pair")


def test_zoom_downsample():
    """downsample_2x(zoom_2x(g)) == g"""
    g = [[1, 2], [3, 4]]
    zoomed = inv.zoom_2x(g)
    assert inv._size(zoomed) == (4, 4)
    back = inv.downsample_2x(zoomed)
    assert inv._grids_equal(back, g), f"downsample∘zoom != identity"
    print("  ✅ test_zoom_downsample")


def test_find_simple_inverse_depth1():
    """If chain is [rotate_90], inverter should find inverse chain [rotate_90]
    (because INVERSE_OPS['rotate_90'] = rotate_270, and that undoes rotate_90)."""
    # Build task: output = rotate_90(input) (base_fn = identity)
    inputs = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[0, 1], [2, 3]]]
    outputs = [inv.rotate_90(g) for g in inputs]
    train = [{"input": inputs[i], "output": outputs[i]} for i in range(3)]
    found = inv.find_inverse_chain(train, max_depth=1, require_identity_match=True)
    assert found == ["rotate_90"], f"Expected ['rotate_90'], got {found}"
    print(f"  ✅ test_find_simple_inverse_depth1 (found {found})")


def test_find_chain_depth2():
    """Chain = [rotate_90, flip_h]. Inverter should find ['flip_h', 'rotate_90']
    when applied to output gives back input."""
    # output = flip_h(rotate_90(input))
    # To invert: apply inv(flip_h) then inv(rotate_90)
    # = flip_h then rotate_270
    # INVERSE_OPS['flip_horizontal'] = flip_horizontal, INVERSE_OPS['rotate_90'] = rotate_270
    # So inverse_chain = ['flip_horizontal', 'rotate_90'] (apply inverses in reverse order)
    inputs = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]], [[3, 4, 5], [6, 7, 8]]]
    outputs = [inv.flip_horizontal(inv.rotate_90(g)) for g in inputs]
    train = [{"input": inputs[i], "output": outputs[i]} for i in range(3)]
    found = inv.find_inverse_chain(train, max_depth=2, require_identity_match=True)
    assert found is not None, "Inverter must find SOME chain of depth ≤ 2"
    # Verify chain actually works
    for ex in train:
        unwrapped = inv._apply_chain(ex["output"], found)
        assert inv._grids_equal(unwrapped, ex["input"]), \
            f"Chain {found} doesn't unwrap correctly"
    print(f"  ✅ test_find_chain_depth2 (found {found})")


def test_no_chain_for_complex_base_fn():
    """If base_fn is NOT identity (e.g., color shift), inverter should NOT
    find an identity-match chain — task isn't pure chain."""
    inputs = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    # Each output: add 1 to each cell (this is NOT in our op vocabulary)
    outputs = [[[c + 1 for c in row] for row in g] for g in inputs]
    train = [{"input": inputs[i], "output": outputs[i]} for i in range(2)]
    found = inv.find_inverse_chain(train, max_depth=3, require_identity_match=True)
    # Should be None because no op chain can undo "+1 to each cell"
    assert found is None, f"Expected None for color shift task, got {found}"
    print(f"  ✅ test_no_chain_for_complex_base_fn (got {found})")


def test_unwrap_train_examples():
    """unwrap_train_examples applies inverse_chain to each output."""
    train = [{"input": [[1, 2]], "output": [[2, 1]]}]   # output = flip_h
    unwrapped = inv.unwrap_train_examples(train, ["flip_horizontal"])
    assert unwrapped == [{"input": [[1, 2]], "output": [[1, 2]]}]
    print("  ✅ test_unwrap_train_examples")


def test_forward_chain_application():
    """If inverse_chain undoes original chain, applying forward should reproduce it."""
    g = [[1, 2, 3], [4, 5, 6]]
    # Suppose inverse_chain is ['flip_horizontal', 'rotate_90']
    # which means we undid (rotate_90 then flip_h) to get back to g
    # So forward should reconstruct: g → rotate_90 → flip_h
    original_chain = inv.flip_horizontal(inv.rotate_90(g))
    inverse_chain = ["flip_horizontal", "rotate_90"]
    forward_fns = inv.forward_chain_from_inverse(inverse_chain)
    reconstructed = inv.apply_forward_chain(g, forward_fns)
    assert inv._grids_equal(reconstructed, original_chain), \
        f"Forward chain doesn't reconstruct: {reconstructed} != {original_chain}"
    print("  ✅ test_forward_chain_application")


def main():
    print("Running Phase 3 chain inverter tests:")
    test_self_inverse_ops()
    test_rotate_pair()
    test_zoom_downsample()
    test_find_simple_inverse_depth1()
    test_find_chain_depth2()
    test_no_chain_for_complex_base_fn()
    test_unwrap_train_examples()
    test_forward_chain_application()
    print("\n✅ All Phase 3 chain inverter tests passed")


if __name__ == "__main__":
    main()
