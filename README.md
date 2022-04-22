# matrix

matrix using const generics (only for nightly)

```rust
    let left = Matrix::<2, 3, u32>([1, 2, 3, 4, 5, 6]);
    let right = Matrix::<2, 3, u32>([1, 2, 3, 4, 5, 6]);
    let sum = left + right;
    assert_eq!(sum, Matrix::<2, 3, u32>([2, 4, 6, 8, 10, 12]));

    let left = Matrix::<2, 3, u32>([3, 7, 2, 2, 4, 3]);
    let right = Matrix::<3, 3, u32>([2, 1, 4, 9, 2, 7, 8, 3, 2]);
    let prd = left * right;
    assert_eq!(prd, Matrix::<2, 3, u32>([85, 23, 65, 64, 19, 42]));
```