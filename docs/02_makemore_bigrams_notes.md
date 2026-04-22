# Lecture 2: Bigram Model

A bigram model looks at one character and predicts the next one.

For a word like `.emma.`, it learns transitions such as `.` -> `e`, `e` -> `m`,
`m` -> `m`, `m` -> `a`, and `a` -> `.`.

That is the whole idea: current character in, next-character probabilities out.
