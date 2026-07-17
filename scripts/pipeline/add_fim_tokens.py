#!/usr/bin/env python3
"""
Add FIM (Fill-in-the-Middle) special tokens to the SentencePiece tokenizer.

Appends 4 special tokens — <PRE>, <SUF>, <MID>, <EOT> — to the existing
4K Unigram model by modifying the protobuf in-place. This preserves ALL
existing token IDs (0–3999) so the pretrained checkpoint is not broken;
the new tokens get IDs 4000–4003.

The model embedding table is then resized separately (resize_embeddings.py)
to 4016 (next multiple of 16), with the 4 FIM rows mean-initialized and
12 padding rows zero-initialized.

Token naming follows Code Llama (<PRE>/<SUF>/<MID>/<EOT>), which Continue.dev
and llama.cpp both template natively.

Usage:
    python3 scripts/pipeline/add_fim_tokens.py \
        --input tokenizer/basque_unigram_4000.model \
        --output tokenizer/basque_unigram_fim.model
"""

import argparse
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

# FIM tokens in PSM/SPM order. <EOT> is the end-of-turn / generation stop.
# These match Code Llama naming (Continue.dev AutocompleteTemplate supports it).
FIM_TOKENS = ["<PRE>", "<SUF>", "<MID>", "<EOT>"]


def add_fim_tokens(input_path: str, output_path: str):
    """Append FIM tokens to a SentencePiece model protobuf."""
    # Load the model as a protobuf
    with open(input_path, "rb") as f:
        model_proto = sp_pb2.ModelProto()
        model_proto.ParseFromString(f.read())

    original_size = len(model_proto.pieces)
    print(f"Original vocab size: {original_size}")

    # Show the existing special tokens and last few pieces for sanity
    print("\nExisting special tokens:")
    for i, piece in enumerate(model_proto.pieces):
        if piece.type != sp_pb2.ModelProto.SentencePiece.NORMAL:
            print(f"  [{i}] {piece.piece!r}  type={piece.type}  score={piece.score}")

    # Check none of the FIM tokens already exist
    existing_pieces = {p.piece for p in model_proto.pieces}
    for tok in FIM_TOKENS:
        if tok in existing_pieces:
            raise ValueError(f"Token {tok!r} already exists in the tokenizer!")

    # Append FIM tokens as USER_DEFINED type
    # USER_DEFINED: never split, treated as atomic, not produced by the model
    # (only inserted programmatically). Score 0.0 (irrelevant for USER_DEFINED).
    print(f"\nAppending {len(FIM_TOKENS)} FIM tokens:")
    for tok in FIM_TOKENS:
        new_piece = model_proto.pieces.add()
        new_piece.piece = tok
        new_piece.score = 0.0
        new_piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
        new_id = len(model_proto.pieces) - 1
        print(f"  [{new_id}] {tok}  type=USER_DEFINED")

    new_size = len(model_proto.pieces)
    print(f"\nNew vocab size: {new_size} (+{new_size - original_size})")

    # Save
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"Saved: {output_path}")

    # ── Verification ──
    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)

    sp_old = spm.SentencePieceProcessor()
    sp_old.Load(input_path)

    sp_new = spm.SentencePieceProcessor()
    sp_new.Load(output_path)

    # 1. Vocab sizes
    print(f"\nOld vocab: {sp_old.get_piece_size()}")
    print(f"New vocab: {sp_new.get_piece_size()}")

    # 2. Existing tokens unchanged (spot-check a range)
    mismatches = 0
    for i in range(original_size):
        if sp_old.id_to_piece(i) != sp_new.id_to_piece(i):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH at {i}: old={sp_old.id_to_piece(i)!r} new={sp_new.id_to_piece(i)!r}")
    print(f"Existing token ID check: {original_size - mismatches}/{original_size} unchanged "
          f"({'✓' if mismatches == 0 else '✗'})")

    # 3. FIM token IDs
    print("\nFIM token IDs:")
    for tok in FIM_TOKENS:
        tid = sp_new.piece_to_id(tok)
        print(f"  {tok:8s} → id {tid}  ({'✓' if tid >= original_size else '✗ unexpected'})")

    # 4. FIM tokens are atomic (appear as single tokens, not split into chars)
    test = "Kaixo<PRE>world<SUF>"
    ids = sp_new.EncodeAsIds(test)
    pieces = sp_new.EncodeAsPieces(test)
    print(f"\nAtomicity test: {test!r}")
    print(f"  pieces: {pieces}")
    print(f"  ids:    {ids}")
    pre_id = sp_new.piece_to_id("<PRE>")
    suf_id = sp_new.piece_to_id("<SUF>")
    assert pre_id in ids, "<PRE> should be a single atomic token!"
    assert suf_id in ids, "<SUF> should be a single atomic token!"
    print(f"  ✓ <PRE> (id {pre_id}) and <SUF> (id {suf_id}) are atomic (not split)")

    # 5. Regular text still tokenizes identically
    sample = "Euskal Herriko hizkuntza ofiziala da euskara"
    old_ids = sp_old.EncodeAsIds(sample)
    new_ids = sp_new.EncodeAsIds(sample)
    print(f"\nRegression test: {sample!r}")
    print(f"  old ids: {old_ids}")
    print(f"  new ids: {new_ids}")
    print(f"  {'✓ identical' if old_ids == new_ids else '✗ DIFFERENT!'}")

    # 6. The new .vocab file
    vocab_path = str(Path(output_path).with_suffix(".vocab"))
    with open(vocab_path, "w", encoding="utf-8") as f:
        for piece in model_proto.pieces:
            f.write(f"{piece.piece}\t{piece.score}\n")
    print(f"\nWrote .vocab: {vocab_path}")

    print("\n" + "=" * 50)
    print("✓ FIM tokenizer ready. Next: resize_embeddings.py")
    print(f"  Tokenizer:  {output_path}")
    print(f"  Vocab:      {original_size} → {new_size}")
    print(f"  Pad target: {((new_size + 15) // 16) * 16} (multiple of 16)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Add FIM tokens to SentencePiece model")
    parser.add_argument("--input", default="tokenizer/basque_unigram_4000.model",
                        help="Input SentencePiece .model")
    parser.add_argument("--output", default="tokenizer/basque_unigram_fim.model",
                        help="Output SentencePiece .model with FIM tokens")
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input tokenizer not found: {args.input}")

    add_fim_tokens(args.input, args.output)


if __name__ == "__main__":
    main()
