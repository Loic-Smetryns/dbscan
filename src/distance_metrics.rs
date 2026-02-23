pub fn mean_absolute_error_1d(a: &f64, b: &f64) -> f64{
    (a-b).abs()
}

pub fn mean_absolute_error_2d((a1, a2): &(f64, f64), (b1, b2): &(f64, f64)) -> f64{
    ((a1-b1).abs() + (a2-b2).abs())/2.
}

pub fn mean_absolute_error_3d((a1, a2, a3): &(f64, f64, f64), (b1, b2, b3): &(f64, f64, f64)) -> f64{
    ((a1-b1).abs() + (a2-b2).abs() + (a3-b3).abs())/3.
}

pub fn mean_absolute_error_nd(x : &[f64], y : &[f64]) -> f64{
    assert!(x.len() == y.len(), "Les tailles des slices doivent être identique.");
    assert!(x.len() > 0, "Les tailles des slices doivent être strictement positives.");
    
    x.iter().zip(y.iter()).map(|(a, b)|{ (a - b).abs() }).sum::<f64>()/(x.len() as f64)
}

pub fn root_mean_square_deviation_1d(a: &f64, b: &f64) -> f64{
    (a-b).abs()
}

pub fn root_mean_square_deviation_2d((a1, a2): &(f64, f64), (b1, b2): &(f64, f64)) -> f64{
    let (r1, r2) = ((a1-b1), (a2-b2));

    ((r1*r1 + r2*r2)/2.).sqrt()
}

pub fn root_mean_square_deviation_3d((a1, a2, a3): &(f64, f64, f64), (b1, b2, b3): &(f64, f64, f64)) -> f64{
    let (r1, r2, r3) = ((a1-b1), (a2-b2), (a3-b3));

    ((r1*r1 + r2*r2 + r3*r3)/3.).sqrt()
}

pub fn root_mean_square_deviation_nd(x : &[f64], y : &[f64]) -> f64{
    assert!(x.len() == y.len(), "Les tailles des slices doivent être identique.");
    assert!(x.len() > 0, "Les tailles des slices doivent être strictement positives.");
    
    (x.iter().zip(y.iter())
    .map(|(a, b)|{ 
        let r = a-b;
        r*r 
    }).sum::<f64>()/(x.len() as f64)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_absolute_error_1d() {
        let result = mean_absolute_error_1d(&0.0, &0.0);
        assert_eq!(result, 0.0);

        let result = mean_absolute_error_1d(&5.0, &5.0);
        assert_eq!(result, 0.0);

        let result = mean_absolute_error_1d(&-2.0, &-2.0);
        assert_eq!(result, 0.0);

        let r1 = mean_absolute_error_1d(&5.0, &2.0);
        let r2 = mean_absolute_error_1d(&2.0, &5.0);
        assert_eq!(r1, 3.0);
        assert_eq!(r2, 3.0);

        let r1 = mean_absolute_error_1d(&-5.0, &2.0);
        let r2 = mean_absolute_error_1d(&2.0, &-5.0);
        assert_eq!(r1, 7.0);
        assert_eq!(r2, 7.0);

        let r1 = mean_absolute_error_1d(&-5.0, &-2.0);
        let r2 = mean_absolute_error_1d(&-2.0, &-5.0);
        assert_eq!(r1, 3.0);
        assert_eq!(r2, 3.0);
    }

    #[test]
    fn test_mean_absolute_error_2d() {
        let (x, y) = ((0.0, 0.0), (0.0, 0.0));

        let r1 = mean_absolute_error_2d(&x, &y);
        assert_eq!(r1, 0.0);

        let (x, y) = ((1.0, 2.0), (0.0, 0.0));

        let r1 = mean_absolute_error_2d(&x, &y);
        let r2 = mean_absolute_error_2d(&y, &x);
        assert_eq!(r1, 1.5);
        assert_eq!(r2, 1.5);

        let (x, y) = ((1.0, 3.0), (4.0, 5.0));

        let r1 = mean_absolute_error_2d(&x, &y);
        let r2 = mean_absolute_error_2d(&y, &x);
        assert_eq!(r1, 2.5);
        assert_eq!(r2, 2.5);

        let (x, y) = ((-1.0, -3.0), (4.0, 5.0));

        let r1 = mean_absolute_error_2d(&x, &y);
        let r2 = mean_absolute_error_2d(&y, &x);
        assert_eq!(r1, 6.5);
        assert_eq!(r2, 6.5);

        let (x, y) = ((-1.0, 3.0), (-4.0, 5.0));

        let r1 = mean_absolute_error_2d(&x, &y);
        let r2 = mean_absolute_error_2d(&y, &x);
        assert_eq!(r1, 2.5);
        assert_eq!(r2, 2.5);

        let (x, y) = ((-1.0, -3.0), (-4.0, -5.0));

        let r1 = mean_absolute_error_2d(&x, &y);
        let r2 = mean_absolute_error_2d(&y, &x);
        assert_eq!(r1, 2.5);
        assert_eq!(r2, 2.5);
    }

    #[test]
    fn test_mean_absolute_error_3d() {
        let (x, y) = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0));

        let r1 = mean_absolute_error_3d(&x, &y);
        assert_eq!(r1, 0.0);

        let (x, y) = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0));

        let r1 = mean_absolute_error_3d(&x, &y);
        let r2 = mean_absolute_error_3d(&y, &x);
        assert_eq!(r1, 2.0);
        assert_eq!(r2, 2.0);

        let (x, y) = ((1.0, 3.0, 7.0), (4.0, 5.0, 6.0));

        let r1 = mean_absolute_error_3d(&x, &y);
        let r2 = mean_absolute_error_3d(&y, &x);
        assert_eq!(r1, 2.0);
        assert_eq!(r2, 2.0);

        let (x, y) = ((-1.0, -3.0, -7.0), (4.0, 5.0, 6.0));

        let r1 = mean_absolute_error_3d(&x, &y);
        let r2 = mean_absolute_error_3d(&y, &x);
        assert_eq!(r1, 26./3.);
        assert_eq!(r2, 26./3.);

        let (x, y) = ((-1.0, 3.0, 7.0), (-4.0, 5.0, 6.0));

        let r1 = mean_absolute_error_3d(&x, &y);
        let r2 = mean_absolute_error_3d(&y, &x);
        assert_eq!(r1, 2.0);
        assert_eq!(r2, 2.0);

        let (x, y) = ((-1.0, -3.0, -7.0), (-4.0, -5.0, -6.0));

        let r1 = mean_absolute_error_3d(&x, &y);
        let r2 = mean_absolute_error_3d(&y, &x);
        assert_eq!(r1, 2.0);
        assert_eq!(r2, 2.0);
    }

    #[test]
    fn test_mean_absolute_error_nd() {
        let x = [0.0];
        let y = [0.0];
        let result = mean_absolute_error_nd(&x, &y);
        assert_eq!(result, 0.0);

        let x = [5.0];
        let y = [2.0];
        let r1 = mean_absolute_error_nd(&x, &y);
        let r2 = mean_absolute_error_nd(&y, &x);
        assert_eq!(r1, 3.0);
        assert_eq!(r2, 3.0);

        let x = [-5.0];
        let y = [2.0];
        let r1 = mean_absolute_error_nd(&x, &y);
        let r2 = mean_absolute_error_nd(&y, &x);
        assert_eq!(r1, 7.0);
        assert_eq!(r2, 7.0);

        let x = [0.0, 0.0];
        let y = [0.0, 0.0];
        let result = mean_absolute_error_nd(&x, &y);
        assert_eq!(result, 0.0);

        let x = [1.0, 2.0];
        let y = [0.0, 0.0];
        let r1 = mean_absolute_error_nd(&x, &y);
        let r2 = mean_absolute_error_nd(&y, &x);
        assert_eq!(r1, 1.5);
        assert_eq!(r2, 1.5);

        let x = [-1.0, -3.0];
        let y = [4.0, 5.0];
        let r1 = mean_absolute_error_nd(&x, &y);
        let r2 = mean_absolute_error_nd(&y, &x);
        assert_eq!(r1, 6.5);
        assert_eq!(r2, 6.5);

        let x = [0.0, 0.0, 0.0];
        let y = [0.0, 0.0, 0.0];
        let result = mean_absolute_error_nd(&x, &y);
        assert_eq!(result, 0.0);

        let x = [1.0, 2.0, 3.0];
        let y = [0.0, 0.0, 0.0];
        let r1 = mean_absolute_error_nd(&x, &y);
        let r2 = mean_absolute_error_nd(&y, &x);
        assert_eq!(r1, 2.0);
        assert_eq!(r2, 2.0);

        let x = [-1.0, -3.0, -7.0];
        let y = [4.0, 5.0, 6.0];
        let r1 = mean_absolute_error_nd(&x, &y);
        let r2 = mean_absolute_error_nd(&y, &x);
        assert_eq!(r1, 26./3.);
        assert_eq!(r2, 26./3.);

        let x: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
        let y = [0.0, 0.0, 0.0, 0.0];
        let result = mean_absolute_error_nd(&x, &y);
        assert_eq!(result, 0.0);

        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 0.0, 0.0, 0.0];
        let r1 = mean_absolute_error_nd(&x, &y);
        let r2 = mean_absolute_error_nd(&y, &x);
        assert_eq!(r1, 2.5);
        assert_eq!(r2, 2.5);

        let x = [-1.0, -2.0, -3.0, -4.0];
        let y = [4.0, 5.0, 6.0, 7.0];
        let r1 = mean_absolute_error_nd(&x, &y);
        let r2 = mean_absolute_error_nd(&y, &x);
        assert_eq!(r1, 8.0);
        assert_eq!(r2, 8.0);
    }

    #[test]
    fn test_root_mean_square_deviation_1d() {
        let result = root_mean_square_deviation_1d(&0.0, &0.0);
        assert_eq!(result, 0.0);

        let result = root_mean_square_deviation_1d(&5.0, &5.0);
        assert_eq!(result, 0.0);

        let result = root_mean_square_deviation_1d(&-2.0, &-2.0);
        assert_eq!(result, 0.0);

        let r1 = root_mean_square_deviation_1d(&5.0, &2.0);
        let r2 = root_mean_square_deviation_1d(&2.0, &5.0);
        assert_eq!(r1, 3.0);
        assert_eq!(r2, 3.0);

        let r1 = root_mean_square_deviation_1d(&-5.0, &2.0);
        let r2 = root_mean_square_deviation_1d(&2.0, &-5.0);
        assert_eq!(r1, 7.0);
        assert_eq!(r2, 7.0);

        let r1 = root_mean_square_deviation_1d(&-5.0, &-2.0);
        let r2 = root_mean_square_deviation_1d(&-2.0, &-5.0);
        assert_eq!(r1, 3.0);
        assert_eq!(r2, 3.0);
    }

    #[test]
    fn test_root_mean_square_deviation_2d() {
        let (x, y) = ((0.0, 0.0), (0.0, 0.0));
        let r1 = root_mean_square_deviation_2d(&x, &y);
        assert_eq!(r1, 0.0);

        let (x, y) = ((1.0, 2.0), (0.0, 0.0));
        let r1 = root_mean_square_deviation_2d(&x, &y);
        let r2 = root_mean_square_deviation_2d(&y, &x);
        assert_eq!(r1, 2.5_f64.sqrt());
        assert_eq!(r2, 2.5_f64.sqrt());

        let (x, y) = ((1.0, 3.0), (4.0, 5.0));
        let r1 = root_mean_square_deviation_2d(&x, &y);
        let r2 = root_mean_square_deviation_2d(&y, &x);
        assert_eq!(r1, 6.5_f64.sqrt());
        assert_eq!(r2, 6.5_f64.sqrt());

        let (x, y) = ((-1.0, -3.0), (4.0, 5.0));
        let r1 = root_mean_square_deviation_2d(&x, &y);
        let r2 = root_mean_square_deviation_2d(&y, &x);
        assert_eq!(r1, 44.5_f64.sqrt());
        assert_eq!(r2, 44.5_f64.sqrt());
    }

    #[test]
    fn test_root_mean_square_deviation_3d() {
        let (x, y) = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        assert_eq!(root_mean_square_deviation_3d(&x, &y), 0.0);

        let (x, y) = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0));
        assert_eq!(root_mean_square_deviation_3d(&x, &y), ((14.0_f64)/3.).sqrt());

        let (x, y) = ((1.0, 3.0, 7.0), (4.0, 5.0, 6.0));
        assert_eq!(root_mean_square_deviation_3d(&x, &y), ((14.0_f64)/3.).sqrt());

        let (x, y) = ((-1.0, -3.0, -7.0), (4.0, 5.0, 6.0));
        assert_eq!(root_mean_square_deviation_3d(&x, &y), ((258_f64)/3.).sqrt());

        let (x, y) = ((-1.0, 3.0, 7.0), (-4.0, 5.0, 6.0));
        assert_eq!(root_mean_square_deviation_3d(&x, &y), ((14.0_f64)/3.).sqrt());

        let (x, y) = ((-1.0, -3.0, -7.0), (-1.0, -3.0, -7.0));
        assert_eq!(root_mean_square_deviation_3d(&x, &y), 0.0);
    }

    #[test]
    fn test_root_mean_square_deviation_nd() {
        let x = [5.0];
        let y = [2.0];
        assert_eq!(root_mean_square_deviation_nd(&x, &y), 3.0);

        let x = [1.0, 2.0, 3.0];
        let y = [0.0, 0.0, 0.0];
        assert_eq!(root_mean_square_deviation_nd(&x, &y), (14.0_f64/3.).sqrt());

        let x = [-1.0, -2.0, -3.0];
        let y = [0.0, 0.0, 0.0];
        assert_eq!(root_mean_square_deviation_nd(&x, &y), (14.0_f64/3.).sqrt());

        let x = [-1.0, 2.0, -3.0];
        let y = [1.0, -2.0, 3.0];
        assert_eq!(root_mean_square_deviation_nd(&x, &y), (56.0_f64/3.).sqrt());

        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(root_mean_square_deviation_nd(&x, &y), 11.0_f64.sqrt());

        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0];
        assert_eq!(root_mean_square_deviation_nd(&x, &y), 0.0);

        let x = [1.0001, 2.0001, 3.0001];
        let y = [1.0, 2.0, 3.0];
        assert!((root_mean_square_deviation_nd(&x, &y)-1e-4).abs() <= 1e-10);
    }
}