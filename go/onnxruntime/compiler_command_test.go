package onnxruntime

import (
	"fmt"
	"slices"
	"testing"
)

// splitCompilerCommand follows cmd/internal/quoted.Split: only leading quotes
// delimit fields, and quoted contents are not unescaped.
func splitCompilerCommand(command string) ([]string, error) {
	isSpace := func(c byte) bool {
		return c == ' ' || c == '\t' || c == '\n' || c == '\r'
	}
	var fields []string
	for len(command) > 0 {
		for len(command) > 0 && isSpace(command[0]) {
			command = command[1:]
		}
		if len(command) == 0 {
			break
		}
		if command[0] == '"' || command[0] == '\'' {
			quote := command[0]
			command = command[1:]
			i := 0
			for i < len(command) && command[i] != quote {
				i++
			}
			if i == len(command) {
				return nil, fmt.Errorf("unterminated %c string", quote)
			}
			fields = append(fields, command[:i])
			command = command[i+1:]
			continue
		}
		i := 0
		for i < len(command) && !isSpace(command[i]) {
			i++
		}
		fields = append(fields, command[:i])
		command = command[i:]
	}
	return fields, nil
}

func TestSplitCompilerCommand(t *testing.T) {
	tests := []struct {
		name    string
		command string
		want    []string
		wantErr bool
	}{
		{name: "plain", command: "gcc", want: []string{"gcc"}},
		{name: "multiple flags", command: " \tgcc -O2\r\n-pthread ", want: []string{"gcc", "-O2", "-pthread"}},
		{name: "quoted path", command: `"/opt/compiler tools/gcc" "-DNAME=hello world"`, want: []string{"/opt/compiler tools/gcc", "-DNAME=hello world"}},
		{name: "single quotes", command: `'/opt/compiler tools/gcc' '-DNAME=hello world'`, want: []string{"/opt/compiler tools/gcc", "-DNAME=hello world"}},
		{name: "unterminated double quote", command: `gcc "-O2`, wantErr: true},
		{name: "unterminated single quote", command: `gcc '-O2`, wantErr: true},
		{name: "empty", command: " \t\r\n"},
		{name: "empty argument", command: `gcc ""`, want: []string{"gcc", ""}},
		{name: "literal backslash", command: `'C:\compiler tools\gcc'`, want: []string{`C:\compiler tools\gcc`}},
		{name: "embedded quote", command: `gcc -DNAME="hello"`, want: []string{"gcc", `-DNAME="hello"`}},
		{name: "adjacent fields", command: `"gcc"'-O2'`, want: []string{"gcc", "-O2"}},
		{name: "nonseparator whitespace", command: "gcc\vflag\fflag", want: []string{"gcc\vflag\fflag"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := splitCompilerCommand(tt.command)
			if (err != nil) != tt.wantErr {
				t.Fatalf("splitCompilerCommand(%q) error = %v, want error %v", tt.command, err, tt.wantErr)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("splitCompilerCommand(%q) = %q, want %q", tt.command, got, tt.want)
			}
		})
	}
}
