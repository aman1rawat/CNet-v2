# CNet-v2
<b>
The second (and hopefully better) version of a library made purely in C for creating and training Artificial Neural Networks.<br>
I will be updating this readme file with necessary explanations and some small snippets of theory whenever necessary.
</b>
<br><br>
Feedbacks and suggestions [Discussions] are highly appreciated :) <br>

<hr>

<h2>Matrix Structure</h2>
Most of our numerical data will be stored in matrices (2D arrays)<br>
We will use heap-allcoated arrays, but not in a pointer-to-pointer manner.<br>
Instead we will have a single pointer pointing to an array of size 'rows X cols'.

```
	struct Matrix{
		int row, col;
		float *data;
	};
```

This will give us:
> -->Better memory locality and perfromance<br>
> -->Reduction overhead of allocating mutliple pointers<br>
> -->Simpler memory management<br>
