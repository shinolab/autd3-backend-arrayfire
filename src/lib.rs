#![allow(unknown_lints)]
#![allow(clippy::manual_slice_size_calculation)]

use std::collections::HashMap;

use arrayfire::*;

use autd3_core::{
    acoustics::{
        directivity::{Directivity, Sphere},
        propagate,
    },
    gain::BitVec,
    geometry::Geometry,
};
use autd3_gain_holo::{
    Complex, HoloError, LinAlgBackend, MatrixX, MatrixXc, Trans, VectorX, VectorXc,
};

pub type AFBackend = arrayfire::Backend;
pub type AFDeviceInfo = (String, String, String, String);

fn convert(trans: Trans) -> MatProp {
    match trans {
        Trans::NoTrans => MatProp::NONE,
        Trans::Trans => MatProp::TRANS,
        Trans::ConjTrans => MatProp::CTRANS,
    }
}

pub struct ArrayFireBackend<D: Directivity> {
    _phantom: std::marker::PhantomData<D>,
}

impl ArrayFireBackend<Sphere> {
    pub fn get_available_backends() -> Vec<AFBackend> {
        arrayfire::get_available_backends()
    }

    pub fn set_backend(backend: AFBackend) {
        arrayfire::set_backend(backend);
    }

    pub fn set_device(device: i32) {
        arrayfire::set_device(device);
    }

    pub fn get_available_devices() -> Vec<AFDeviceInfo> {
        let cur_dev = arrayfire::get_device();
        let r = (0..arrayfire::device_count())
            .map(|i| {
                arrayfire::set_device(i);
                arrayfire::device_info()
            })
            .collect();
        arrayfire::set_device(cur_dev);
        r
    }
}

impl Default for ArrayFireBackend<Sphere> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<D: Directivity> ArrayFireBackend<D> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: Directivity> LinAlgBackend<D> for ArrayFireBackend<D> {
    type MatrixXc = Array<c32>;
    type MatrixX = Array<f32>;
    type VectorXc = Array<c32>;
    type VectorX = Array<f32>;

    fn generate_propagation_matrix(
        &self,
        geometry: &Geometry,
        foci: &[autd3_core::geometry::Point3],
        filter: Option<&HashMap<usize, BitVec>>,
    ) -> Result<Self::MatrixXc, HoloError> {
        let g = if let Some(filter) = filter {
            geometry
                .devices()
                .flat_map(|dev| {
                    dev.iter().filter_map(move |tr| {
                        if let Some(filter) = filter.get(&dev.idx()) {
                            if filter[tr.idx()] {
                                Some(foci.iter().map(move |fp| {
                                    propagate::<D>(tr, dev.wavenumber(), dev.axial_direction(), fp)
                                }))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                })
                .flatten()
                .collect::<Vec<_>>()
        } else {
            geometry
                .devices()
                .flat_map(|dev| {
                    dev.iter().flat_map(move |tr| {
                        foci.iter().map(move |fp| {
                            propagate::<D>(tr, dev.wavenumber(), dev.axial_direction(), fp)
                        })
                    })
                })
                .collect::<Vec<_>>()
        };

        unsafe {
            Ok(Array::new(
                std::slice::from_raw_parts(g.as_ptr() as *const c32, g.len()),
                Dim4::new(&[foci.len() as u64, (g.len() / foci.len()) as _, 1, 1]),
            ))
        }
    }

    fn alloc_v(&self, size: usize) -> Result<Self::VectorX, HoloError> {
        Ok(Array::new_empty(Dim4::new(&[size as _, 1, 1, 1])))
    }

    fn alloc_m(&self, rows: usize, cols: usize) -> Result<Self::MatrixX, HoloError> {
        Ok(Array::new_empty(Dim4::new(&[rows as _, cols as _, 1, 1])))
    }

    fn alloc_cv(&self, size: usize) -> Result<Self::VectorXc, HoloError> {
        Ok(Array::new_empty(Dim4::new(&[size as _, 1, 1, 1])))
    }

    fn alloc_cm(&self, rows: usize, cols: usize) -> Result<Self::MatrixXc, HoloError> {
        Ok(Array::new_empty(Dim4::new(&[rows as _, cols as _, 1, 1])))
    }

    fn alloc_zeros_v(&self, size: usize) -> Result<Self::VectorX, HoloError> {
        Ok(arrayfire::constant(0., Dim4::new(&[size as _, 1, 1, 1])))
    }

    fn alloc_zeros_cv(&self, size: usize) -> Result<Self::VectorXc, HoloError> {
        Ok(arrayfire::constant(
            c32::new(0., 0.),
            Dim4::new(&[size as _, 1, 1, 1]),
        ))
    }

    fn alloc_zeros_cm(&self, rows: usize, cols: usize) -> Result<Self::MatrixXc, HoloError> {
        Ok(arrayfire::constant(
            c32::new(0., 0.),
            Dim4::new(&[rows as _, cols as _, 1, 1]),
        ))
    }

    fn to_host_v(&self, v: Self::VectorX) -> Result<VectorX, HoloError> {
        let mut r = VectorX::zeros(v.elements());
        v.host(r.as_mut_slice());
        Ok(r)
    }

    fn to_host_m(&self, v: Self::MatrixX) -> Result<MatrixX, HoloError> {
        let mut r = MatrixX::zeros(v.dims()[0] as _, v.dims()[1] as _);
        v.host(r.as_mut_slice());
        Ok(r)
    }

    fn to_host_cv(&self, v: Self::VectorXc) -> Result<VectorXc, HoloError> {
        let n = v.elements();
        let mut r = VectorXc::zeros(n);
        unsafe {
            v.host(std::slice::from_raw_parts_mut(
                r.as_mut_ptr() as *mut c32,
                n,
            ));
        }
        Ok(r)
    }

    fn to_host_cm(&self, v: Self::MatrixXc) -> Result<MatrixXc, HoloError> {
        let n = v.elements();
        let mut r = MatrixXc::zeros(v.dims()[0] as _, v.dims()[1] as _);
        unsafe {
            v.host(std::slice::from_raw_parts_mut(
                r.as_mut_ptr() as *mut c32,
                n,
            ));
        }
        Ok(r)
    }

    fn from_slice_v(&self, v: &[f32]) -> Result<Self::VectorX, HoloError> {
        Ok(Array::new(v, Dim4::new(&[v.len() as _, 1, 1, 1])))
    }

    fn from_slice_m(
        &self,
        rows: usize,
        cols: usize,
        v: &[f32],
    ) -> Result<Self::MatrixX, HoloError> {
        Ok(Array::new(v, Dim4::new(&[rows as _, cols as _, 1, 1])))
    }

    fn from_slice_cv(&self, v: &[f32]) -> Result<Self::VectorXc, HoloError> {
        let r = Array::new(v, Dim4::new(&[v.len() as _, 1, 1, 1]));
        Ok(arrayfire::cplx(&r))
    }

    fn from_slice2_cv(&self, r: &[f32], i: &[f32]) -> Result<Self::VectorXc, HoloError> {
        let r = Array::new(r, Dim4::new(&[r.len() as _, 1, 1, 1]));
        let i = Array::new(i, Dim4::new(&[i.len() as _, 1, 1, 1]));
        Ok(arrayfire::cplx2(&r, &i, false).cast())
    }

    fn from_slice2_cm(
        &self,
        rows: usize,
        cols: usize,
        r: &[f32],
        i: &[f32],
    ) -> Result<Self::MatrixXc, HoloError> {
        let r = Array::new(r, Dim4::new(&[rows as _, cols as _, 1, 1]));
        let i = Array::new(i, Dim4::new(&[rows as _, cols as _, 1, 1]));
        Ok(arrayfire::cplx2(&r, &i, false).cast())
    }

    fn copy_from_slice_v(&self, v: &[f32], dst: &mut Self::VectorX) -> Result<(), HoloError> {
        let n = v.len();
        if n == 0 {
            return Ok(());
        }
        let v = self.from_slice_v(v)?;
        let seqs = [Seq::new(0u32, n as u32 - 1, 1)];
        arrayfire::assign_seq(dst, &seqs, &v);
        Ok(())
    }

    fn copy_to_v(&self, src: &Self::VectorX, dst: &mut Self::VectorX) -> Result<(), HoloError> {
        let seqs = [Seq::new(0u32, src.elements() as u32 - 1, 1)];
        arrayfire::assign_seq(dst, &seqs, src);
        Ok(())
    }

    fn copy_to_m(&self, src: &Self::MatrixX, dst: &mut Self::MatrixX) -> Result<(), HoloError> {
        let seqs = [
            Seq::new(0u32, src.dims()[0] as u32 - 1, 1),
            Seq::new(0u32, src.dims()[1] as u32 - 1, 1),
        ];
        arrayfire::assign_seq(dst, &seqs, src);
        Ok(())
    }

    fn clone_v(&self, v: &Self::VectorX) -> Result<Self::VectorX, HoloError> {
        Ok(v.copy())
    }

    fn clone_m(&self, v: &Self::MatrixX) -> Result<Self::MatrixX, HoloError> {
        Ok(v.copy())
    }

    fn clone_cv(&self, v: &Self::VectorXc) -> Result<Self::VectorXc, HoloError> {
        Ok(v.copy())
    }

    fn clone_cm(&self, v: &Self::MatrixXc) -> Result<Self::MatrixXc, HoloError> {
        Ok(v.copy())
    }

    fn make_complex2_v(
        &self,
        real: &Self::VectorX,
        imag: &Self::VectorX,
        v: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        *v = arrayfire::cplx2(real, imag, false).cast();
        Ok(())
    }

    fn create_diagonal(&self, v: &Self::VectorX, a: &mut Self::MatrixX) -> Result<(), HoloError> {
        *a = arrayfire::diag_create(v, 0);
        Ok(())
    }

    fn create_diagonal_c(
        &self,
        v: &Self::VectorXc,
        a: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        *a = arrayfire::diag_create(v, 0);
        Ok(())
    }

    fn get_diagonal(&self, a: &Self::MatrixX, v: &mut Self::VectorX) -> Result<(), HoloError> {
        *v = arrayfire::diag_extract(a, 0);
        Ok(())
    }

    fn real_cm(&self, a: &Self::MatrixXc, b: &mut Self::MatrixX) -> Result<(), HoloError> {
        *b = arrayfire::real(a);
        Ok(())
    }

    fn imag_cm(&self, a: &Self::MatrixXc, b: &mut Self::MatrixX) -> Result<(), HoloError> {
        *b = arrayfire::imag(a);
        Ok(())
    }

    fn scale_assign_cv(
        &self,
        a: autd3_gain_holo::Complex,
        b: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        let a = c32::new(a.re, a.im);
        *b = arrayfire::mul(b, &a, false);
        Ok(())
    }

    fn conj_assign_v(&self, b: &mut Self::VectorXc) -> Result<(), HoloError> {
        *b = arrayfire::conjg(b);
        Ok(())
    }

    fn exp_assign_cv(&self, v: &mut Self::VectorXc) -> Result<(), HoloError> {
        *v = arrayfire::exp(v);
        Ok(())
    }

    fn concat_col_cm(
        &self,
        a: &Self::MatrixXc,
        b: &Self::MatrixXc,
        c: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        *c = arrayfire::join(1, a, b);
        Ok(())
    }

    fn max_v(&self, m: &Self::VectorX) -> Result<f32, HoloError> {
        Ok(arrayfire::max_all(m).0)
    }

    fn hadamard_product_cm(
        &self,
        x: &Self::MatrixXc,
        y: &Self::MatrixXc,
        z: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        *z = arrayfire::mul(x, y, false);
        Ok(())
    }

    fn dot(&self, x: &Self::VectorX, y: &Self::VectorX) -> Result<f32, HoloError> {
        let r = arrayfire::dot(x, y, MatProp::NONE, MatProp::NONE);
        let mut v = [0.];
        r.host(&mut v);
        Ok(v[0])
    }

    fn dot_c(
        &self,
        x: &Self::VectorXc,
        y: &Self::VectorXc,
    ) -> Result<autd3_gain_holo::Complex, HoloError> {
        let r = arrayfire::dot(x, y, MatProp::CONJ, MatProp::NONE);
        let mut v = [c32::new(0., 0.)];
        r.host(&mut v);
        Ok(autd3_gain_holo::Complex::new(v[0].re, v[0].im))
    }

    fn add_v(&self, alpha: f32, a: &Self::VectorX, b: &mut Self::VectorX) -> Result<(), HoloError> {
        *b = arrayfire::add(&arrayfire::mul(a, &alpha, false), b, false);
        Ok(())
    }

    fn add_m(&self, alpha: f32, a: &Self::MatrixX, b: &mut Self::MatrixX) -> Result<(), HoloError> {
        *b = arrayfire::add(&arrayfire::mul(a, &alpha, false), b, false);
        Ok(())
    }

    fn gevv_c(
        &self,
        trans_a: autd3_gain_holo::Trans,
        trans_b: autd3_gain_holo::Trans,
        alpha: autd3_gain_holo::Complex,
        a: &Self::VectorXc,
        x: &Self::VectorXc,
        beta: autd3_gain_holo::Complex,
        y: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        let alpha = vec![c32::new(alpha.re, alpha.im)];
        let beta = vec![c32::new(beta.re, beta.im)];
        let trans_a = convert(trans_a);
        let trans_b = convert(trans_b);
        arrayfire::gemm(y, trans_a, trans_b, alpha, a, x, beta);
        Ok(())
    }

    fn gemv_c(
        &self,
        trans: autd3_gain_holo::Trans,
        alpha: autd3_gain_holo::Complex,
        a: &Self::MatrixXc,
        x: &Self::VectorXc,
        beta: autd3_gain_holo::Complex,
        y: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        let alpha = vec![c32::new(alpha.re, alpha.im)];
        let beta = vec![c32::new(beta.re, beta.im)];
        let trans = convert(trans);
        arrayfire::gemm(y, trans, MatProp::NONE, alpha, a, x, beta);
        Ok(())
    }

    fn gemm_c(
        &self,
        trans_a: autd3_gain_holo::Trans,
        trans_b: autd3_gain_holo::Trans,
        alpha: autd3_gain_holo::Complex,
        a: &Self::MatrixXc,
        b: &Self::MatrixXc,
        beta: autd3_gain_holo::Complex,
        y: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        let alpha = vec![c32::new(alpha.re, alpha.im)];
        let beta = vec![c32::new(beta.re, beta.im)];
        let trans_a = convert(trans_a);
        let trans_b = convert(trans_b);
        arrayfire::gemm(y, trans_a, trans_b, alpha, a, b, beta);
        Ok(())
    }

    fn solve_inplace(&self, a: &Self::MatrixX, x: &mut Self::VectorX) -> Result<(), HoloError> {
        *x = arrayfire::solve(a, x, MatProp::NONE);
        Ok(())
    }

    fn reduce_col(&self, a: &Self::MatrixX, b: &mut Self::VectorX) -> Result<(), HoloError> {
        *b = arrayfire::sum(a, 1);
        Ok(())
    }

    fn cols_c(&self, m: &Self::MatrixXc) -> Result<usize, HoloError> {
        Ok(m.dims()[1] as _)
    }

    fn scaled_to_cv(
        &self,
        a: &Self::VectorXc,
        b: &Self::VectorXc,
        c: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        let tmp = arrayfire::div(a, &arrayfire::abs(a), false);
        *c = arrayfire::mul(&tmp, b, false);
        Ok(())
    }

    fn scaled_to_assign_cv(
        &self,
        a: &Self::VectorXc,
        b: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        *b = arrayfire::div(b, &arrayfire::abs(b), false);
        *b = arrayfire::mul(a, b, false);
        Ok(())
    }

    fn gen_back_prop(
        &self,
        m: usize,
        n: usize,
        transfer: &Self::MatrixXc,
    ) -> Result<Self::MatrixXc, HoloError> {
        let mut b = self.alloc_zeros_cm(m, n)?;

        let mut tmp = self.alloc_zeros_cm(n, n)?;

        self.gemm_c(
            Trans::NoTrans,
            Trans::ConjTrans,
            Complex::new(1., 0.),
            transfer,
            transfer,
            Complex::new(0., 0.),
            &mut tmp,
        )?;

        let mut denominator = arrayfire::diag_extract(&tmp, 0);
        let a = c32::new(1., 0.);
        denominator = arrayfire::div(&a, &denominator, false);

        self.create_diagonal_c(&denominator, &mut tmp)?;

        self.gemm_c(
            Trans::ConjTrans,
            Trans::NoTrans,
            Complex::new(1., 0.),
            transfer,
            &tmp,
            Complex::new(0., 0.),
            &mut b,
        )?;

        Ok(b)
    }

    fn norm_squared_cv(&self, a: &Self::VectorXc, b: &mut Self::VectorX) -> Result<(), HoloError> {
        *b = arrayfire::abs(a);
        *b = arrayfire::mul(b, b, false);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use autd3::driver::autd3_device::AUTD3;
    use autd3_core::{
        acoustics::directivity::Sphere,
        defined::PI,
        geometry::{Point3, UnitQuaternion},
    };

    use nalgebra::{ComplexField, Normed};

    use autd3_gain_holo::{Amplitude, Pa, Trans};

    use super::*;

    use rand::Rng;

    const N: usize = 10;
    const EPS: f32 = 1e-3;

    fn generate_geometry(size: usize) -> Geometry {
        Geometry::new(
            (0..size)
                .flat_map(|i| {
                    (0..size).map(move |j| {
                        AUTD3 {
                            pos: Point3::new(
                                i as f32 * AUTD3::DEVICE_WIDTH,
                                j as f32 * AUTD3::DEVICE_HEIGHT,
                                0.,
                            ),
                            rot: UnitQuaternion::identity(),
                        }
                        .into()
                    })
                })
                .collect(),
        )
    }

    fn gen_foci(n: usize) -> impl Iterator<Item = (Point3, Amplitude)> {
        (0..n).map(move |i| {
            (
                Point3::new(
                    90. + 10. * (2.0 * PI * i as f32 / n as f32).cos(),
                    70. + 10. * (2.0 * PI * i as f32 / n as f32).sin(),
                    150.,
                ),
                10e3 * Pa,
            )
        })
    }

    fn make_random_v(
        backend: &ArrayFireBackend<Sphere>,
        size: usize,
    ) -> Result<<ArrayFireBackend<Sphere> as LinAlgBackend<Sphere>>::VectorX, HoloError> {
        let mut rng = rand::rng();
        let v: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(size)
            .collect();
        backend.from_slice_v(&v)
    }

    fn make_random_m(
        backend: &ArrayFireBackend<Sphere>,
        rows: usize,
        cols: usize,
    ) -> Result<<ArrayFireBackend<Sphere> as LinAlgBackend<Sphere>>::MatrixX, HoloError> {
        let mut rng = rand::rng();
        let v: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(rows * cols)
            .collect();
        backend.from_slice_m(rows, cols, &v)
    }

    fn make_random_cv(
        backend: &ArrayFireBackend<Sphere>,
        size: usize,
    ) -> Result<<ArrayFireBackend<Sphere> as LinAlgBackend<Sphere>>::VectorXc, HoloError> {
        let mut rng = rand::rng();
        let real: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(size)
            .collect();
        let imag: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(size)
            .collect();
        backend.from_slice2_cv(&real, &imag)
    }

    fn make_random_cm(
        backend: &ArrayFireBackend<Sphere>,
        rows: usize,
        cols: usize,
    ) -> Result<<ArrayFireBackend<Sphere> as LinAlgBackend<Sphere>>::MatrixXc, HoloError> {
        let mut rng = rand::rng();
        let real: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(rows * cols)
            .collect();
        let imag: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(rows * cols)
            .collect();
        backend.from_slice2_cm(rows, cols, &real, &imag)
    }

    #[rstest::fixture]
    fn backend() -> ArrayFireBackend<Sphere> {
        ArrayFireBackend::set_backend(AFBackend::CPU);
        ArrayFireBackend {
            _phantom: std::marker::PhantomData,
        }
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let v = backend.alloc_v(N)?;
        let v = backend.to_host_v(v)?;

        assert_eq!(N, v.len());
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_m(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let m = backend.alloc_m(N, 2 * N)?;
        let m = backend.to_host_m(m)?;

        assert_eq!(N, m.nrows());
        assert_eq!(2 * N, m.ncols());
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let v = backend.alloc_cv(N)?;
        let v = backend.to_host_cv(v)?;

        assert_eq!(N, v.len());
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_cm(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let m = backend.alloc_cm(N, 2 * N)?;
        let m = backend.to_host_cm(m)?;

        assert_eq!(N, m.nrows());
        assert_eq!(2 * N, m.ncols());
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_zeros_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let v = backend.alloc_zeros_v(N)?;
        let v = backend.to_host_v(v)?;

        assert_eq!(N, v.len());
        assert!(v.iter().all(|&v| v == 0.));
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_zeros_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let v = backend.alloc_zeros_cv(N)?;
        let v = backend.to_host_cv(v)?;

        assert_eq!(N, v.len());
        assert!(v.iter().all(|&v| v == Complex::new(0., 0.)));
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_zeros_cm(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let m = backend.alloc_zeros_cm(N, 2 * N)?;
        let m = backend.to_host_cm(m)?;

        assert_eq!(N, m.nrows());
        assert_eq!(2 * N, m.ncols());
        assert!(m.iter().all(|&v| v == Complex::new(0., 0.)));
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_cols_c(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let m = backend.alloc_cm(N, 2 * N)?;

        assert_eq!(2 * N, backend.cols_c(&m)?);

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let rng = rand::rng();

        let v: Vec<f32> = rng
            .sample_iter(rand::distr::StandardUniform)
            .take(N)
            .collect();

        let c = backend.from_slice_v(&v)?;
        let c = backend.to_host_v(c)?;

        assert_eq!(N, c.len());
        v.iter().zip(c.iter()).for_each(|(&r, &c)| {
            assert_eq!(r, c);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice_m(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let rng = rand::rng();

        let v: Vec<f32> = rng
            .sample_iter(rand::distr::StandardUniform)
            .take(N * 2 * N)
            .collect();

        let c = backend.from_slice_m(N, 2 * N, &v)?;
        let c = backend.to_host_m(c)?;

        assert_eq!(N, c.nrows());
        assert_eq!(2 * N, c.ncols());
        (0..2 * N).for_each(|col| {
            (0..N).for_each(|row| {
                assert_eq!(v[col * N + row], c[(row, col)]);
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let rng = rand::rng();

        let real: Vec<f32> = rng
            .sample_iter(rand::distr::StandardUniform)
            .take(N)
            .collect();

        let c = backend.from_slice_cv(&real)?;
        let c = backend.to_host_cv(c)?;

        assert_eq!(N, c.len());
        real.iter().zip(c.iter()).for_each(|(r, c)| {
            assert_eq!(r, &c.re);
            assert_eq!(0.0, c.im);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice2_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let mut rng = rand::rng();

        let real: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(N)
            .collect();
        let imag: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(N)
            .collect();

        let c = backend.from_slice2_cv(&real, &imag)?;
        let c = backend.to_host_cv(c)?;

        assert_eq!(N, c.len());
        real.iter()
            .zip(imag.iter())
            .zip(c.iter())
            .for_each(|((r, i), c)| {
                assert_eq!(r, &c.re);
                assert_eq!(i, &c.im);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice2_cm(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let mut rng = rand::rng();

        let real: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(N * 2 * N)
            .collect();
        let imag: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(N * 2 * N)
            .collect();

        let c = backend.from_slice2_cm(N, 2 * N, &real, &imag)?;
        let c = backend.to_host_cm(c)?;

        assert_eq!(N, c.nrows());
        assert_eq!(2 * N, c.ncols());
        (0..2 * N).for_each(|col| {
            (0..N).for_each(|row| {
                assert_eq!(real[col * N + row], c[(row, col)].re);
                assert_eq!(imag[col * N + row], c[(row, col)].im);
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_copy_from_slice_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        {
            let mut a = backend.alloc_zeros_v(N)?;
            let mut rng = rand::rng();
            let v = (&mut rng)
                .sample_iter(rand::distr::StandardUniform)
                .take(N / 2)
                .collect::<Vec<f32>>();

            backend.copy_from_slice_v(&v, &mut a)?;

            let a = backend.to_host_v(a)?;
            (0..N / 2).for_each(|i| {
                assert_eq!(v[i], a[i]);
            });
            (N / 2..N).for_each(|i| {
                assert_eq!(0., a[i]);
            });
        }

        {
            let mut a = backend.alloc_zeros_v(N)?;
            let v = [];

            backend.copy_from_slice_v(&v, &mut a)?;

            let a = backend.to_host_v(a)?;
            a.iter().for_each(|&a| {
                assert_eq!(0., a);
            });
        }

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_copy_to_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_v(&backend, N)?;
        let mut b = backend.alloc_v(N)?;

        backend.copy_to_v(&a, &mut b)?;

        let a = backend.to_host_v(a)?;
        let b = backend.to_host_v(b)?;
        a.iter().zip(b.iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_copy_to_m(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_m(&backend, N, N)?;
        let mut b = backend.alloc_m(N, N)?;

        backend.copy_to_m(&a, &mut b)?;

        let a = backend.to_host_m(a)?;
        let b = backend.to_host_m(b)?;
        a.iter().zip(b.iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let c = make_random_v(&backend, N)?;
        let c2 = backend.clone_v(&c)?;

        let c = backend.to_host_v(c)?;
        let c2 = backend.to_host_v(c2)?;

        c.iter().zip(c2.iter()).for_each(|(c, c2)| {
            assert_eq!(c, c2);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone_m(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let c = make_random_m(&backend, N, N)?;
        let c2 = backend.clone_m(&c)?;

        let c = backend.to_host_m(c)?;
        let c2 = backend.to_host_m(c2)?;

        c.iter().zip(c2.iter()).for_each(|(c, c2)| {
            assert_eq!(c, c2);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let c = make_random_cv(&backend, N)?;
        let c2 = backend.clone_cv(&c)?;

        let c = backend.to_host_cv(c)?;
        let c2 = backend.to_host_cv(c2)?;

        c.iter().zip(c2.iter()).for_each(|(c, c2)| {
            assert_eq!(c.re, c2.re);
            assert_eq!(c.im, c2.im);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone_cm(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let c = make_random_cm(&backend, N, N)?;
        let c2 = backend.clone_cm(&c)?;

        let c = backend.to_host_cm(c)?;
        let c2 = backend.to_host_cm(c2)?;

        c.iter().zip(c2.iter()).for_each(|(c, c2)| {
            assert_eq!(c.re, c2.re);
            assert_eq!(c.im, c2.im);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_make_complex2_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let real = make_random_v(&backend, N)?;
        let imag = make_random_v(&backend, N)?;

        let mut c = backend.alloc_cv(N)?;
        backend.make_complex2_v(&real, &imag, &mut c)?;

        let real = backend.to_host_v(real)?;
        let imag = backend.to_host_v(imag)?;
        let c = backend.to_host_cv(c)?;
        real.iter()
            .zip(imag.iter())
            .zip(c.iter())
            .for_each(|((r, i), c)| {
                assert_eq!(r, &c.re);
                assert_eq!(i, &c.im);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_diagonal(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let diagonal = make_random_v(&backend, N)?;

        let mut c = backend.alloc_m(N, N)?;

        backend.create_diagonal(&diagonal, &mut c)?;

        let diagonal = backend.to_host_v(diagonal)?;
        let c = backend.to_host_m(c)?;
        (0..N).for_each(|i| {
            (0..N).for_each(|j| {
                if i == j {
                    assert_eq!(diagonal[i], c[(i, j)]);
                } else {
                    assert_eq!(0.0, c[(i, j)]);
                }
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_diagonal_c(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let diagonal = make_random_cv(&backend, N)?;

        let mut c = backend.alloc_cm(N, N)?;

        backend.create_diagonal_c(&diagonal, &mut c)?;

        let diagonal = backend.to_host_cv(diagonal)?;
        let c = backend.to_host_cm(c)?;
        (0..N).for_each(|i| {
            (0..N).for_each(|j| {
                if i == j {
                    assert_eq!(diagonal[i].re, c[(i, j)].re);
                    assert_eq!(diagonal[i].im, c[(i, j)].im);
                } else {
                    assert_eq!(0.0, c[(i, j)].re);
                    assert_eq!(0.0, c[(i, j)].im);
                }
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_get_diagonal(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let m = make_random_m(&backend, N, N)?;
        let mut diagonal = backend.alloc_v(N)?;

        backend.get_diagonal(&m, &mut diagonal)?;

        let m = backend.to_host_m(m)?;
        let diagonal = backend.to_host_v(diagonal)?;
        (0..N).for_each(|i| {
            assert_eq!(m[(i, i)], diagonal[i]);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_norm_squared_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let v = make_random_cv(&backend, N)?;

        let mut abs = backend.alloc_v(N)?;
        backend.norm_squared_cv(&v, &mut abs)?;

        let v = backend.to_host_cv(v)?;
        let abs = backend.to_host_v(abs)?;
        v.iter().zip(abs.iter()).for_each(|(v, abs)| {
            assert_approx_eq::assert_approx_eq!(v.norm_squared(), abs, EPS);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_real_cm(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let v = make_random_cm(&backend, N, N)?;
        let mut r = backend.alloc_m(N, N)?;

        backend.real_cm(&v, &mut r)?;

        let v = backend.to_host_cm(v)?;
        let r = backend.to_host_m(r)?;
        (0..N).for_each(|i| {
            (0..N).for_each(|j| {
                assert_approx_eq::assert_approx_eq!(v[(i, j)].re, r[(i, j)], EPS);
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_imag_cm(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let v = make_random_cm(&backend, N, N)?;
        let mut r = backend.alloc_m(N, N)?;

        backend.imag_cm(&v, &mut r)?;

        let v = backend.to_host_cm(v)?;
        let r = backend.to_host_m(r)?;
        (0..N).for_each(|i| {
            (0..N).for_each(|j| {
                assert_approx_eq::assert_approx_eq!(v[(i, j)].im, r[(i, j)], EPS);
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scale_assign_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let mut v = make_random_cv(&backend, N)?;
        let vc = backend.clone_cv(&v)?;
        let mut rng = rand::rng();
        let scale = Complex::new(rng.random(), rng.random());

        backend.scale_assign_cv(scale, &mut v)?;

        let v = backend.to_host_cv(v)?;
        let vc = backend.to_host_cv(vc)?;
        v.iter().zip(vc.iter()).for_each(|(&v, &vc)| {
            assert_approx_eq::assert_approx_eq!(scale * vc, v, EPS);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_conj_assign_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let mut v = make_random_cv(&backend, N)?;
        let vc = backend.clone_cv(&v)?;

        backend.conj_assign_v(&mut v)?;

        let v = backend.to_host_cv(v)?;
        let vc = backend.to_host_cv(vc)?;
        v.iter().zip(vc.iter()).for_each(|(&v, &vc)| {
            assert_eq!(vc.re, v.re);
            assert_eq!(vc.im, -v.im);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_exp_assign_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let mut v = make_random_cv(&backend, N)?;
        let vc = backend.clone_cv(&v)?;

        backend.exp_assign_cv(&mut v)?;

        let v = backend.to_host_cv(v)?;
        let vc = backend.to_host_cv(vc)?;
        v.iter().zip(vc.iter()).for_each(|(v, vc)| {
            assert_approx_eq::assert_approx_eq!(vc.exp(), v, EPS);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_concat_col_cm(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_cm(&backend, N, N)?;
        let b = make_random_cm(&backend, N, 2 * N)?;
        let mut c = backend.alloc_cm(N, N + 2 * N)?;

        backend.concat_col_cm(&a, &b, &mut c)?;

        let a = backend.to_host_cm(a)?;
        let b = backend.to_host_cm(b)?;
        let c = backend.to_host_cm(c)?;
        (0..N).for_each(|col| (0..N).for_each(|row| assert_eq!(a[(row, col)], c[(row, col)])));
        (0..2 * N)
            .for_each(|col| (0..N).for_each(|row| assert_eq!(b[(row, col)], c[(row, N + col)])));
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_max_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let v = make_random_v(&backend, N)?;

        let max = backend.max_v(&v)?;

        let v = backend.to_host_v(v)?;
        assert_eq!(
            *v.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            max
        );
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_hadamard_product_cm(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_cm(&backend, N, N)?;
        let b = make_random_cm(&backend, N, N)?;
        let mut c = backend.alloc_cm(N, N)?;

        backend.hadamard_product_cm(&a, &b, &mut c)?;

        let a = backend.to_host_cm(a)?;
        let b = backend.to_host_cm(b)?;
        let c = backend.to_host_cm(c)?;
        c.iter()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((c, a), b)| {
                assert_approx_eq::assert_approx_eq!(a.re * b.re - a.im * b.im, c.re, EPS);
                assert_approx_eq::assert_approx_eq!(a.re * b.im + a.im * b.re, c.im, EPS);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dot(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_v(&backend, N)?;
        let b = make_random_v(&backend, N)?;

        let dot = backend.dot(&a, &b)?;

        let a = backend.to_host_v(a)?;
        let b = backend.to_host_v(b)?;
        let expect = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>();
        assert_approx_eq::assert_approx_eq!(dot, expect, EPS);
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dot_c(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_cv(&backend, N)?;
        let b = make_random_cv(&backend, N)?;

        let dot = backend.dot_c(&a, &b)?;

        let a = backend.to_host_cv(a)?;
        let b = backend.to_host_cv(b)?;
        let expect = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex>();
        assert_approx_eq::assert_approx_eq!(dot.re, expect.re, EPS);
        assert_approx_eq::assert_approx_eq!(dot.im, expect.im, EPS);
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_add_v(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_v(&backend, N)?;
        let mut b = make_random_v(&backend, N)?;
        let bc = backend.clone_v(&b)?;

        let mut rng = rand::rng();
        let alpha = rng.random();

        backend.add_v(alpha, &a, &mut b)?;

        let a = backend.to_host_v(a)?;
        let b = backend.to_host_v(b)?;
        let bc = backend.to_host_v(bc)?;
        b.iter()
            .zip(a.iter())
            .zip(bc.iter())
            .for_each(|((b, a), bc)| {
                assert_approx_eq::assert_approx_eq!(alpha * a + bc, b, EPS);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_add_m(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_m(&backend, N, N)?;
        let mut b = make_random_m(&backend, N, N)?;
        let bc = backend.clone_m(&b)?;

        let mut rng = rand::rng();
        let alpha = rng.random();

        backend.add_m(alpha, &a, &mut b)?;

        let a = backend.to_host_m(a)?;
        let b = backend.to_host_m(b)?;
        let bc = backend.to_host_m(bc)?;
        b.iter()
            .zip(a.iter())
            .zip(bc.iter())
            .for_each(|((b, a), bc)| {
                assert_approx_eq::assert_approx_eq!(alpha * a + bc, b, EPS);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_gevv_c(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let mut rng = rand::rng();

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gevv_c(Trans::NoTrans, Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cv(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b.transpose() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gevv_c(
                Trans::NoTrans,
                Trans::ConjTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cv(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b.adjoint() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, 1, 1)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gevv_c(Trans::Trans, Trans::NoTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cv(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.transpose() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, 1, 1)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gevv_c(
                Trans::ConjTrans,
                Trans::NoTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cv(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.adjoint() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_gemv_c(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let m = N;
        let n = 2 * N;

        let mut rng = rand::rng();

        {
            let a = make_random_cm(&backend, m, n)?;
            let b = make_random_cv(&backend, n)?;
            let mut c = make_random_cv(&backend, m)?;
            let cc = backend.clone_cv(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemv_c(Trans::NoTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cv(c)?;
            let cc = backend.to_host_cv(cc)?;
            let expected = a * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, n, m)?;
            let b = make_random_cv(&backend, n)?;
            let mut c = make_random_cv(&backend, m)?;
            let cc = backend.clone_cv(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemv_c(Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cv(c)?;
            let cc = backend.to_host_cv(cc)?;
            let expected = a.transpose() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, n, m)?;
            let b = make_random_cv(&backend, n)?;
            let mut c = make_random_cv(&backend, m)?;
            let cc = backend.clone_cv(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemv_c(Trans::ConjTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cv(c)?;
            let cc = backend.to_host_cv(cc)?;
            let expected = a.adjoint() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_gemm_c(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let m = N;
        let n = 2 * N;
        let k = 3 * N;

        let mut rng = rand::rng();

        {
            let a = make_random_cm(&backend, m, k)?;
            let b = make_random_cm(&backend, k, n)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::NoTrans, Trans::NoTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, m, k)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::NoTrans, Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b.transpose() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, m, k)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(
                Trans::NoTrans,
                Trans::ConjTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b.adjoint() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, k, n)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::Trans, Trans::NoTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.transpose() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::Trans, Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.transpose() * b.transpose() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::Trans, Trans::ConjTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.transpose() * b.adjoint() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, k, n)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(
                Trans::ConjTrans,
                Trans::NoTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.adjoint() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::ConjTrans, Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.adjoint() * b.transpose() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(
                Trans::ConjTrans,
                Trans::ConjTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.adjoint() * b.adjoint() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_solve_inplace(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        {
            let tmp = make_random_m(&backend, N, N)?;
            let tmp = backend.to_host_m(tmp)?;

            let a = &tmp * tmp.adjoint();

            let mut rng = rand::rng();
            let x = VectorX::from_iterator(N, (0..N).map(|_| rng.random()));

            let b = &a * &x;

            let aa = backend.from_slice_m(N, N, a.as_slice())?;
            let mut bb = backend.from_slice_v(b.as_slice())?;

            backend.solve_inplace(&aa, &mut bb)?;

            let b2 = &a * backend.to_host_v(bb)?;
            assert!(approx::relative_eq!(b, b2, epsilon = 1e-3));
        }

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_reduce_col(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_m(&backend, N, N)?;

        let mut b = backend.alloc_v(N)?;

        backend.reduce_col(&a, &mut b)?;

        let a = backend.to_host_m(a)?;
        let b = backend.to_host_v(b)?;

        (0..N).for_each(|row| {
            let sum = a.row(row).iter().sum::<f32>();
            assert_approx_eq::assert_approx_eq!(sum, b[row], EPS);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scaled_to_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_cv(&backend, N)?;
        let b = make_random_cv(&backend, N)?;
        let mut c = backend.alloc_cv(N)?;

        backend.scaled_to_cv(&a, &b, &mut c)?;

        let a = backend.to_host_cv(a)?;
        let b = backend.to_host_cv(b)?;
        let c = backend.to_host_cv(c)?;
        c.iter()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((&c, &a), &b)| {
                assert_approx_eq::assert_approx_eq!(c, a / a.abs() * b, EPS);
            });

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scaled_to_assign_cv(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let a = make_random_cv(&backend, N)?;
        let mut b = make_random_cv(&backend, N)?;
        let bc = backend.clone_cv(&b)?;

        backend.scaled_to_assign_cv(&a, &mut b)?;

        let a = backend.to_host_cv(a)?;
        let b = backend.to_host_cv(b)?;
        let bc = backend.to_host_cv(bc)?;
        b.iter()
            .zip(a.iter())
            .zip(bc.iter())
            .for_each(|((&b, &a), &bc)| {
                assert_approx_eq::assert_approx_eq!(b, bc / bc.abs() * a, EPS);
            });

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[case(1, 2)]
    #[case(2, 1)]
    fn test_generate_propagation_matrix(
        #[case] dev_num: usize,
        #[case] foci_num: usize,
        backend: ArrayFireBackend<Sphere>,
    ) -> Result<(), HoloError> {
        let reference = |geometry: Geometry, foci: Vec<Point3>| {
            let mut g = MatrixXc::zeros(
                foci.len(),
                geometry
                    .iter()
                    .map(|dev| dev.num_transducers())
                    .sum::<usize>(),
            );
            let transducers = geometry
                .iter()
                .flat_map(|dev| dev.iter().map(|tr| (dev.idx(), tr)))
                .collect::<Vec<_>>();
            (0..foci.len()).for_each(|i| {
                (0..transducers.len()).for_each(|j| {
                    g[(i, j)] = propagate::<Sphere>(
                        transducers[j].1,
                        geometry[transducers[j].0].wavenumber(),
                        geometry[transducers[j].0].axial_direction(),
                        &foci[i],
                    )
                })
            });
            g
        };

        let geometry = generate_geometry(dev_num);
        let foci = gen_foci(foci_num).map(|(p, _)| p).collect::<Vec<_>>();

        let g = backend.generate_propagation_matrix(&geometry, &foci, None)?;
        let g = backend.to_host_cm(g)?;
        reference(geometry, foci)
            .iter()
            .zip(g.iter())
            .for_each(|(r, g)| {
                assert_approx_eq::assert_approx_eq!(r.re, g.re, EPS);
                assert_approx_eq::assert_approx_eq!(r.im, g.im, EPS);
            });

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[case(1, 2)]
    #[case(2, 1)]
    fn test_generate_propagation_matrix_with_filter(
        #[case] dev_num: usize,
        #[case] foci_num: usize,
        backend: ArrayFireBackend<Sphere>,
    ) -> Result<(), HoloError> {
        use std::collections::HashMap;

        let filter = |geometry: &Geometry| {
            geometry
                .iter()
                .map(|dev| {
                    let mut filter = BitVec::new();
                    dev.iter().for_each(|tr| {
                        filter.push(tr.idx() > dev.num_transducers() / 2);
                    });
                    (dev.idx(), filter)
                })
                .collect::<HashMap<_, _>>()
        };

        let reference = |geometry, foci: Vec<Point3>| {
            let filter = filter(&geometry);
            let transducers = geometry
                .iter()
                .flat_map(|dev| {
                    dev.iter().filter_map(|tr| {
                        if filter[&dev.idx()][tr.idx()] {
                            Some((dev.idx(), tr))
                        } else {
                            None
                        }
                    })
                })
                .collect::<Vec<_>>();

            let mut g = MatrixXc::zeros(foci.len(), transducers.len());
            (0..foci.len()).for_each(|i| {
                (0..transducers.len()).for_each(|j| {
                    g[(i, j)] = propagate::<Sphere>(
                        transducers[j].1,
                        geometry[transducers[j].0].wavenumber(),
                        geometry[transducers[j].0].axial_direction(),
                        &foci[i],
                    )
                })
            });
            g
        };

        let geometry = generate_geometry(dev_num);
        let foci = gen_foci(foci_num).map(|(p, _)| p).collect::<Vec<_>>();
        let filter = filter(&geometry);

        let g = backend.generate_propagation_matrix(&geometry, &foci, Some(&filter))?;
        let g = backend.to_host_cm(g)?;
        assert_eq!(g.nrows(), foci.len());
        assert_eq!(
            g.ncols(),
            geometry
                .iter()
                .map(|dev| dev.num_transducers() / 2)
                .sum::<usize>()
        );
        reference(geometry, foci)
            .iter()
            .zip(g.iter())
            .for_each(|(r, g)| {
                assert_approx_eq::assert_approx_eq!(r.re, g.re, EPS);
                assert_approx_eq::assert_approx_eq!(r.im, g.im, EPS);
            });

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    fn test_gen_back_prop(backend: ArrayFireBackend<Sphere>) -> Result<(), HoloError> {
        let geometry = generate_geometry(1);
        let foci = gen_foci(2).map(|(p, _)| p).collect::<Vec<_>>();

        let m = geometry
            .iter()
            .map(|dev| dev.num_transducers())
            .sum::<usize>();
        let n = foci.len();

        let g = backend.generate_propagation_matrix(&geometry, &foci, None)?;

        let b = backend.gen_back_prop(m, n, &g)?;
        let g = backend.to_host_cm(g)?;
        let reference = {
            let mut b = MatrixXc::zeros(m, n);
            (0..n).for_each(|i| {
                let x = 1.0 / g.rows(i, 1).iter().map(|x| x.norm_sqr()).sum::<f32>();
                (0..m).for_each(|j| {
                    b[(j, i)] = g[(i, j)].conj() * x;
                })
            });
            b
        };

        let b = backend.to_host_cm(b)?;
        reference.iter().zip(b.iter()).for_each(|(r, b)| {
            assert_approx_eq::assert_approx_eq!(r.re, b.re, EPS);
            assert_approx_eq::assert_approx_eq!(r.im, b.im, EPS);
        });
        Ok(())
    }
}
