README.md

 am getting the following error:

AssertionError: Batches returned wrong contents. For example, input sequence 1 in the first batch was [640 641 642 643 644]

Any hint what’s wrong? I think batches[0,0,x] in the test code does not match [640 641 642 643 644] but I don’t understand why?

However my output for the following parameters matches with the example given:

int_text = np.arange(1,19)
batch_size = 3
seq_length = 2
batches = get_batches(int_text, batch_size, seq_length)
print(batches)

Output:

[[[[ 1 2]
[ 7 8]
[13 14]]

[[ 2 3]
[ 8 9]
[14 15]]]

[[[ 3 4]
[ 9 10]
[15 16]]

[[ 4 5]
[10 11]
[16 17]]]

[[[ 5 6]
[11 12]
[17 18]]

[[ 6 7]
[12 13]
[18 1]]]]
