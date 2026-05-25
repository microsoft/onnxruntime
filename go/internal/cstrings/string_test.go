package cstrings

import "testing"

func TestCStringToString(t *testing.T) {
	if s := CStringToString(nil); s != "" {
		t.Errorf("nil should return empty string, got %q", s)
	}

	b := []byte("hello\x00world")
	s := CStringToString(&b[0])
	if s != "hello" {
		t.Errorf("expected 'hello', got %q", s)
	}
}

func TestStringToCBytes(t *testing.T) {
	b := StringToCBytes("test")
	if len(b) != 5 {
		t.Errorf("expected length 5, got %d", len(b))
	}
	if b[4] != 0 {
		t.Error("last byte should be null terminator")
	}
	if string(b[:4]) != "test" {
		t.Errorf("expected 'test', got %q", string(b[:4]))
	}
}

func TestCStringRoundtrip(t *testing.T) {
	orig := "hello world"
	b := StringToCBytes(orig)
	back := CStringToString(&b[0])
	if back != orig {
		t.Errorf("roundtrip failed: %q -> %q", orig, back)
	}
}
