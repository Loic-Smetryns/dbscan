pub mod distance_metrics;

use std::usize;
use std::collections::HashMap;

pub const OUTLIER : usize = usize::MAX;

pub struct DbscanModel<T: Clone, F: Fn(&T, &T) -> f64> {
    n_clusters: usize,
    samples: Vec<T>,
    labels: Vec<usize>,
    distance_metric: F
}

#[derive(Clone, PartialEq)]
enum VisitState{ 
    Visited,
    Unvisited 
}

#[derive(Clone)]
struct SampleInfo{
    state: VisitState,
    neighbors: Vec<usize>,
    core: bool
}

impl<T: Clone, F: Fn(&T, &T) -> f64> DbscanModel<T, F> {
    fn new(dataset: &[T], distance_metric: F) -> Self{
        Self {
            n_clusters: 0,
            samples: dataset.to_vec(),
            labels: vec![OUTLIER; dataset.len()],
            distance_metric: distance_metric
        }
    }

    /* Naïve */
    pub fn train(dataset: &[T], distance_metric: F, epsilon: f64, min_neighbor: usize) -> Self {
        assert!(epsilon >= 0.0, "epsilon doit être positif");

        let mut model = Self::new(dataset, distance_metric);
        let mut samples_info = vec![
            SampleInfo { 
                state: VisitState::Unvisited, 
                neighbors: Vec::with_capacity(32), 
                core: false 
            };
            dataset.len()
        ];

        for i in 0..dataset.len(){
            let neighbors: Vec<usize> = (i+1..dataset.len()).filter(
                |&k| (model.distance_metric)(&dataset[i], &dataset[k]) <= epsilon
            ).collect();

            samples_info[i].neighbors.extend(&neighbors);
            for &k in &neighbors {
                samples_info[k].neighbors.push(i);
            }

            samples_info[i].core = samples_info[i].neighbors.len() + 1 >= min_neighbor;
        }

        let mut stack = Vec::<usize>::with_capacity(32);
        for i in 0..samples_info.len(){
            if samples_info[i].state == VisitState::Unvisited && samples_info[i].core{
                samples_info[i].state = VisitState::Visited;
                stack.push(i);

                while let Some(current) = stack.pop() {
                    model.labels[current] = model.n_clusters;

                    if samples_info[current].core {
                        for j in 0..samples_info[current].neighbors.len(){
                            let i_neighbor=samples_info[current].neighbors[j];
                            
                            if samples_info[i_neighbor].state == VisitState::Unvisited{
                                samples_info[i_neighbor].state = VisitState::Visited;
                                stack.push(i_neighbor);
                            }
                        }
                    }
                }

                model.n_clusters += 1;
            }
        }

        model
    }

    pub fn predict(&self, x: &T, n_neighbor: usize)->usize{
        if self.n_clusters > 0 {
            let mut nearest_neighbors = Vec::with_capacity(n_neighbor+1);

            for (i, neighbor) in self.samples.iter().enumerate() {
                if self.labels[i] == OUTLIER { continue; }
                
                let d = (self.distance_metric)(x, neighbor);

                nearest_neighbors.push((d, self.labels[i]));

                if nearest_neighbors.len() > n_neighbor{
                    nearest_neighbors.remove(
                        nearest_neighbors.iter().enumerate().max_by(
                            |(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Greater)
                        ).map(|(i, _)|{ i }).unwrap()
                    );
                }
            }

            let mut count: HashMap<usize, usize> = HashMap::new();

            for (_, label) in nearest_neighbors{
                *count.entry(label).or_insert(0) += 1;
            }

            *count.iter().max_by(|(_, v), (_, v2)| { v.cmp(v2) }).unwrap().0
        }
        else{
            OUTLIER
        }
    }

    pub fn predict_with_epsilon(&self, x: &T, epsilon: f64, n_neighbor: usize)->usize{
        if self.n_clusters > 0 {
            let mut nearest_neighbors = Vec::with_capacity(n_neighbor+1);

            for (i, neighbor) in self.samples.iter().enumerate() {
                if self.labels[i] == OUTLIER { continue; }
                
                let d = (self.distance_metric)(x, neighbor);

                nearest_neighbors.push((d, self.labels[i]));

                if nearest_neighbors.len() > n_neighbor{
                    nearest_neighbors.remove(
                        nearest_neighbors.iter().enumerate().max_by(
                            |(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Greater)
                        ).map(|(i, _)|{ i }).unwrap()
                    );
                }
            }

            let f_nearest_neighbors = nearest_neighbors.iter().filter(|(d, _)| { *d <= epsilon });

            if f_nearest_neighbors.clone().count() == 0{
                return OUTLIER;
            }

            let mut count: HashMap<usize, usize> = HashMap::new();

            for (_, label) in f_nearest_neighbors{
                *count.entry(*label).or_insert(0) += 1;
            }

            *count.iter().max_by(|(_, v), (_, v2)| { v.cmp(v2) }).unwrap().0
        }
        else{
            OUTLIER
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dbscan_with_empty() {
        let dataset = vec![];

        let model = DbscanModel::train(&dataset, distance_metrics::mean_absolute_error_1d, 0.2, 1);

        assert_eq!(model.n_clusters, 0);
        assert_eq!(model.labels.is_empty(), true);

        let prediction_1 = model.predict(&0.0, 1);
        assert_eq!(prediction_1, OUTLIER);

        let prediction_2 = model.predict(&1.0, 1);
        assert_eq!(prediction_2, OUTLIER);
    }

    #[test]
    fn dbscan_with_epsilon_0() {
        let dataset = vec![0., 0.1, 0.2, -0.3, -0.2, 0.4, -1., 1., 1.1, 1.2, 1.3, 1.4, 1.9, 9.0, 2.0];

        let model = DbscanModel::train(&dataset, distance_metrics::mean_absolute_error_1d, 0.0, 0);

        assert_eq!(model.n_clusters, 15);
        assert_eq!(model.labels, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);


        let prediction_1 = model.predict(&0.002, 1);
        assert_eq!(prediction_1, 0);

        let prediction_2 = model.predict(&1.91, 1);
        assert_eq!(prediction_2, 12);

        let prediction_3 = model.predict(&9.0, 1);
        assert_eq!(prediction_3, 13);

        let prediction_4 = model.predict(&-0.31, 1);
        assert_eq!(prediction_4, 3);
    }

    #[test]
    fn dbscan_with_floats() {
        let dataset = vec![0., 0.1, 0.2, -0.3, -0.2, 0.4, -1., 1., 1.1, 1.2, 1.3, 1.4, 1.9, 9.0, 2.0];

        let model = DbscanModel::train(&dataset, distance_metrics::mean_absolute_error_1d, 0.2, 2);

        assert_eq!(model.n_clusters, 3);
        assert_eq!(model.labels, [0, 0, 0, 0, 0, 0, OUTLIER, 1, 1, 1, 1, 1, 2, OUTLIER, 2]);

        let prediction_1 = model.predict(&-0.05, 3);
        assert_eq!(prediction_1, 0);

        let prediction_2 = model.predict(&1.66, 3);
        assert_eq!(prediction_2, 2);

        let prediction_3 = model.predict(&9.0, 1);
        assert_eq!(prediction_3, 2);

        let prediction_4 = model.predict(&1.25, 3);
        assert_eq!(prediction_4, 1);
    }

    #[test]
    fn dbscan_with_moon_1() {
        let dataset = vec![
            ( 0.079,  0.108 ), ( 0.156,  1.364 ), ( 0.325,  1.350 ), ( 2.274, -0.585 ), ( 1.370,  0.380 ),
            ( 0.016,  0.266 ), ( 1.643, -0.594 ), ( 0.399, -0.298 ), (-1.338,  0.375 ), ( 1.314,  0.807 ),
            ( 0.655, -0.696 ), ( 0.942, -0.577 ), (-0.072,  0.604 ), ( 0.980, -0.523 ), (-1.047,  1.001 ),
            ( 0.834,  1.119 ), (-0.339,  1.363 ), ( 2.619, -0.157 ), ( 2.868,  0.548 ), ( 0.266,  0.014 ),
            (-0.481,  1.308 ), ( 0.322, -0.083 ), ( 0.519, -0.365 ), (-0.627,  1.073 ), ( 2.246, -0.491 ),
            (-1.208,  0.628 ), ( 2.341, -0.222 ), ( 0.816,  1.197 ), (-0.123,  1.386 ), ( 0.058,  0.727 ),
            ( 1.083,  0.996 ), (-0.895,  1.225 ), ( 1.875, -0.543 ), ( 2.951,  0.777 ), ( 1.293,  0.198 ),
            ( 0.113,  1.314 ), ( 1.241, -0.853 ), (-1.485,  0.072 ), ( 1.099,  0.954 ), (-1.421,  0.134 ),
            ( 1.194, -0.692 ), (-1.341,  0.515 ), (-1.103,  0.742 ), ( 0.477,  1.299 ), ( 1.749, -0.810 ),
            ( 2.721,  0.126 ), ( 1.341,  0.500 ), ( 1.395,  0.074 ), ( 2.779,  0.348 ), ( 2.494, -0.161 )
        ];

        let model = DbscanModel::train(&dataset, distance_metrics::root_mean_square_deviation_2d, 0.4, 2);
        
        assert_eq!(model.n_clusters, 2);
        assert_eq!(model.labels, [
            0, 1, 1, 0, 1, 
            0, 0, 0, 1, 1, 
            0, 0, 0, 0, 1, 
            1, 1, 0, 0, 0, 
            1, 0, 0, 1, 0, 
            1, 0, 1, 1, 0, 
            1, 1, 0, 0, 1, 
            1, 0, 1, 1, 1, 
            0, 1, 1, 1, 0, 
            0, 1, 1, 0, 0, 
        ]);

        let prediction_1 = model.predict(&(0.056, 0.076), 3);
        assert_eq!(prediction_1, 0);

        let prediction_2 = model.predict(&(0.110, 0.965), 3);
        assert_eq!(prediction_2, 1);

        let prediction_3 = model.predict(&(1.608, -0.414), 3);
        assert_eq!(prediction_3, 0);

        let prediction_4 = model.predict(&(-0.946, 0.265), 3);
        assert_eq!(prediction_4, 1);
    }

    #[test]
    fn dbscan_with_moon_2() {
        let dataset = vec![
            ( 0.056,  0.076 ), ( 0.110,  0.965 ), ( 0.230,  0.954 ), ( 1.608, -0.414 ), ( 0.969,  0.269 ),
            ( 0.011,  0.188 ), ( 1.161, -0.420 ), ( 0.282, -0.211 ), (-0.946,  0.265 ), ( 0.929,  0.571 ),
            ( 0.463, -0.492 ), ( 0.666, -0.408 ), (-0.051,  0.427 ), ( 0.693, -0.370 ), (-0.741,  0.708 ),
            ( 0.590,  0.791 ), (-0.240,  0.964 ), ( 1.851, -0.111 ), ( 2.028,  0.387 ), ( 0.188,  0.010 ),
            (-0.340,  0.925 ), ( 0.228, -0.059 ), ( 0.367, -0.258 ), (-0.444,  0.758 ), ( 1.588, -0.347 ),
            (-0.855,  0.444 ), ( 1.656, -0.157 ), ( 0.577,  0.846 ), (-0.087,  0.980 ), ( 0.041,  0.514 ),
            ( 0.766,  0.704 ), (-0.633,  0.865 ), ( 1.325, -0.384 ), ( 2.088,  0.549 ), ( 0.914,  0.140 ),
            ( 0.080,  0.929 ), ( 0.878, -0.603 ), (-1.050,  0.051 ), ( 0.777,  0.674 ), (-1.004,  0.095 ),
            ( 0.844, -0.489 ), (-0.948,  0.364 ), (-0.780,  0.525 ), ( 0.337,  0.918 ), ( 1.237, -0.573 ),
            ( 1.924,  0.089 ), ( 0.948,  0.354 ), ( 0.987,  0.052 ), ( 1.965,  0.246 ), ( 1.764, -0.114 )
        ];

        let model = DbscanModel::train(&dataset, distance_metrics::root_mean_square_deviation_2d, 0.14142, 3);
        
        assert_eq!(model.n_clusters, 9);
        assert_eq!(model.labels, [
            0, 1, 1, OUTLIER, 2, 
            0, 3, 0, 4, 6, 
            OUTLIER, 5, OUTLIER, 5, 4, 
            6, 1, 8, 7, 0, 
            1, 0, 0, 1, OUTLIER, 
            4, 8, 6, 1, OUTLIER, 
            6, 4, 3, 7, 2, 
            1, 5, 4, 6, 4, 
            5, 4, 4, 1, 3, 
            7, 2, 2, 7, 8,  
        ]);

        let prediction_1 = model.predict(&(0.056, 0.076), 3);
        assert_eq!(prediction_1, 0);

        let prediction_2 = model.predict(&(0.110, 0.965), 3);
        assert_eq!(prediction_2, 1);

        let prediction_3 = model.predict(&(1.608, -0.414), 3);
        assert_eq!(prediction_3, 8);

        let prediction_4 = model.predict(&(-0.946, 0.265), 3);
        assert_eq!(prediction_4, 4);
    }
}

use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

#[pyclass]
#[pyo3(name="DbscanModel")]
struct PyDbscanModel{
    model: Option<DbscanModel<Box<[f64]>, fn(&Box<[f64]>, &Box<[f64]>) -> f64>>,
    entry_size: usize
}

#[pymethods]
impl PyDbscanModel {
    #[new]
    fn new() -> Self {
        PyDbscanModel {
            model: None,
            entry_size: 0
        }
    }

    fn is_trained(&self) -> bool {
        !self.model.is_none()
    }

    fn train(&mut self, dataset: PyReadonlyArray2<f64>, epsilon: f64, min_neighbor: usize){
        let data = dataset.as_array();

        let n = data.shape()[0];
        let entry_size = data.shape()[1];

        let mut dataset = Vec::<Box<[f64]>>::with_capacity(n*entry_size);

        for y in 0..n{
            let mut array = Vec::with_capacity(entry_size);

            for x in 0..entry_size{
                array.push(*data.get([y, x]).unwrap());
            }

            dataset.push(array.into_boxed_slice());
        }

        self.model = Some(DbscanModel::train(&dataset, |x, y|{distance_metrics::root_mean_square_deviation_nd(x.as_ref(), y.as_ref())}, epsilon, min_neighbor));
        self.entry_size = entry_size;
    }

    fn predict_once(&self, x: PyReadonlyArray1<f64>, n_neighbor: usize)->usize{
        if let Some(model) = &self.model{
            let data = x.as_array();

            let mut array = Vec::with_capacity(self.entry_size);
            for x in 0..self.entry_size{
                array.push(*data.get([x]).unwrap());
            }

            model.predict(&array.into_boxed_slice(), n_neighbor)
        }
        else{
            usize::MAX
        }
    }

    fn predict(&self, py: Python, dataset: PyReadonlyArray2<f64>, n_neighbor: usize) -> Py<PyArray1<usize>> {
        let data = dataset.as_array();
        let n = data.shape()[0];
        let entry_size = data.shape()[1];

        let result_array = PyArray1::<usize>::zeros(py, n, false);
        let slice = unsafe { result_array.as_slice_mut().unwrap() };

        if let Some(model) = &self.model{
            let mut dataset = Vec::<Box<[f64]>>::with_capacity(n*entry_size);

            for y in 0..n{
                let mut array = Vec::with_capacity(entry_size);
    
                for x in 0..entry_size{
                    array.push(*data.get([y, x]).unwrap());
                }
    
                dataset.push(array.into_boxed_slice());
            }
            
            for (i, x) in dataset.iter().enumerate(){
                let z = model.predict(&x, n_neighbor);
                slice[i] = z;
            }
        }
        else{
            for i in 0..n{
                slice[i] = usize::MAX
            }
        }

        result_array.to_owned().into()
    }

    fn predict_once_with_epsilon(&self, x: PyReadonlyArray1<f64>, epsilon: f64, n_neighbor: usize)->usize{
        if let Some(model) = &self.model{
            let data = x.as_array();

            let mut array = Vec::with_capacity(self.entry_size);
            for x in 0..self.entry_size{
                array.push(*data.get([x]).unwrap());
            }

            model.predict_with_epsilon(&array.into_boxed_slice(), epsilon, n_neighbor)
        }
        else{
            usize::MAX
        }
    }

    fn predict_with_epsilon(&self, py: Python, dataset: PyReadonlyArray2<f64>, epsilon: f64, n_neighbor: usize) -> Py<PyArray1<usize>> {
        let data = dataset.as_array();
        let n = data.shape()[0];
        let entry_size = data.shape()[1];

        let result_array = PyArray1::<usize>::zeros(py, n, false);
        let slice = unsafe { result_array.as_slice_mut().unwrap() };

        if let Some(model) = &self.model{
            let mut dataset = Vec::<Box<[f64]>>::with_capacity(n*entry_size);

            for y in 0..n{
                let mut array = Vec::with_capacity(entry_size);
    
                for x in 0..entry_size{
                    array.push(*data.get([y, x]).unwrap());
                }
    
                dataset.push(array.into_boxed_slice());
            }
            
            for (i, x) in dataset.iter().enumerate(){
                let z = model.predict_with_epsilon(&x, epsilon, n_neighbor);
                slice[i] = z;
            }
        }
        else{
            for i in 0..n{
                slice[i] = usize::MAX
            }
        }

        result_array.to_owned().into()
    }
}

#[pymodule(name = "dbscan")]
fn init_module_1(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDbscanModel>()?;
    Ok(())
}