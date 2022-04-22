# matrix

Matrix using stack by definition with const generics (only for nightly)

```rust
    // A + B
    let left = Matrix::<2, 3, u32>([1, 2, 3, 4, 5, 6]);
    let right = Matrix::<2, 3, u32>([1, 2, 3, 4, 5, 6]);
    let sum = left + right;
    assert_eq!(sum, Matrix::<2, 3, u32>([2, 4, 6, 8, 10, 12]));

    // A * B
    let left = Matrix::<2, 3, u32>([3, 7, 2, 2, 4, 3]);
    let right = Matrix::<3, 3, u32>([2, 1, 4, 9, 2, 7, 8, 3, 2]);
    let prd = left * right;
    assert_eq!(prd, Matrix::<2, 3, u32>([85, 23, 65, 64, 19, 42]));

    // A * k
    let left = Matrix::<3, 2, i32>([45, 34, 25, 21, 13, 12]);
    let right = 3;
    assert_eq!(
        left * right,
        Matrix::<3, 2, i32>([135, 102, 75, 63, 39, 36])
    );

    // A - B
    let left = Matrix::<2, 3, i32>([1, 2, 3, 4, 5, 6]);
    let right = Matrix::<2, 3, i32>([6, 5, 4, 3, 2, 1]);
    let diff = left - right;
    assert_eq!(diff, Matrix::<2, 3, i32>([-5, -3, -1, 1, 3, 5]));

    // Solve Ax = b with Gaussian elimination
    //
    // 2a + 2b - 4c + 5d = 16
    //  a +  b +  c +  d = 10
    // -a + 2b - 3c -  d = -2
    //  a + 2b + 3c - 4d = -2
    //
    // (a, b, c, d) = (1, 2, 3, 4)
    let a = Matrix::<4, 4, f64>([
        2.0, 3.0, -4.0, 5.0, 1.0, 1.0, 1.0, 1.0, -1.0, 2.0, -3.0, 1.0, 1.0, 2.0, 3.0, -4.0,
    ]);
    let b = Matrix::<4, 1, f64>([16.0, 10.0, -2.0, -2.0]);
    let x = solve_eqn_gauss(a, b);
    assert!((1.0 - x.0[0]).abs() < 1e-10);
    assert!((2.0 - x.0[1]).abs() < 1e-10);
    assert!((3.0 - x.0[2]).abs() < 1e-10);
    assert!((4.0 - x.0[3]).abs() < 1e-10);

    // LU decomposition
    let matrix = Matrix::<4, 4, f64>([
        2.0, 3.0, -4.0, 5.0, 1.0, 1.0, 1.0, 1.0, -1.0, 2.0, -3.0, 1.0, 1.0, 2.0, 3.0, -4.0,
    ]);
    let (l, u) = matrix.lu_decomposition();
    assert_eq!(l * u, matrix);

    // Solve Ax = b with LU decomposition
    let a = Matrix::<4, 4, f64>([
        2.0, 3.0, -4.0, 5.0, 1.0, 1.0, 1.0, 1.0, -1.0, 2.0, -3.0, 1.0, 1.0, 2.0, 3.0, -4.0,
    ]);
    let b = Matrix::<4, 1, f64>([16.0, 10.0, -2.0, -2.0]);
    let x = solve_eqn(a, b);
    assert!((1.0 - x.0[0]).abs() < 1e-10);
    assert!((2.0 - x.0[1]).abs() < 1e-10);
    assert!((3.0 - x.0[2]).abs() < 1e-10);
    assert!((4.0 - x.0[3]).abs() < 1e-10);

    // transpose
    let matrix = Matrix::<4, 3, u32>([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(
        matrix.transpose(),
        Matrix::<3, 4, u32>([1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12])
    );
```