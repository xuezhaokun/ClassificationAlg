package pp3;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import Jama.Matrix;
import Jama.util.*;

public class GenerativeAlg {
	
	public static double[][] readData(String data_file, String label_file) throws IOException {
		double[][] dataset = ClassificationAlg.readData(data_file);
		double[] labels = ClassificationAlg.readLabels(label_file);
		return ClassificationAlg.combineDataWithLabels(dataset, labels);
	}

	public static List<double[][]> splitDataByClass(double[][] data_with_labels) {
		int dimension = data_with_labels[0].length;
		List<double[][]> data_by_class = new ArrayList<double[][]>();
		List<double[]> data_in_c1 = new ArrayList<double[]>();
		List<double[]> data_in_c2 = new ArrayList<double[]>();
		for (double[] data_record : data_with_labels) {
			if ((int) data_record[dimension - 1] == 1) {
				data_in_c1.addAll(Arrays.asList(data_record));
			} else {
				data_in_c2.addAll(Arrays.asList(data_record));
			}
		}
		data_by_class.add(data_in_c1.toArray(new double[][]{}));
		data_by_class.add(data_in_c2.toArray(new double[][]{}));
		return data_by_class;
	}
	
	public static List<Double> countNs(List<double[][]> data_by_class) {
		List<Double> ns = new ArrayList<Double>();
		double n1 = data_by_class.get(0).length;
		double n2 = data_by_class.get(1).length;
		ns.add(n1);
		ns.add(n2);
		return ns;
	}
	
	public static List<Matrix> calculateMus(List<double[][]> data_by_class) {
		double[][] data_in_c1 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(0));
		double[][] data_in_c2 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(1));
		List<Matrix> mus = new ArrayList<Matrix>();
		List<Double> ns = countNs(data_by_class);
		double n1 = ns.get(0);
		double n2 = ns.get(1);
		Matrix data_c1_matrix = new Matrix(data_in_c1).transpose();
		Matrix data_c2_martix = new Matrix(data_in_c2).transpose();
		double[][] data_in_c1_t = data_c1_matrix.getArray();
		double[][] data_in_c2_t = data_c2_martix.getArray();
		double[] mu1 = new double[data_in_c1_t.length];
		double[] mu2 = new double[data_in_c2_t.length];
		
		for (int i = 0; i < data_in_c1_t.length; i++) {
			double temp = 0;
			for (int j = 0; j < data_in_c1_t[0].length; j++) {
				temp += data_in_c1_t[i][j];
			}
			mu1[i] = temp/n1;
		}
		
		Matrix mu1_Matrix = new Matrix(mu1, 1).transpose();
		mus.add(mu1_Matrix);
		for (int i = 0; i < data_in_c2_t.length; i++) {
			double temp = 0;
			for (int j = 0; j < data_in_c2_t[0].length; j++) {
				temp += data_in_c2_t[i][j];
			}
			mu2[i] = temp/n2;
		}
		Matrix mu2_Matrix = new Matrix(mu2, 1).transpose();
		mus.add(mu2_Matrix);
		return mus;
	}
	
	public static Matrix calculateS(List<double[][]> data_by_class) {
		double[][] data_in_c1 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(0));
		double[][] data_in_c2 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(1));
		int dimension = data_in_c1[0].length;
		List<Double> mus = calculateMus(data_by_class);
		double mu1 = mus.get(0);
		double mu2 = mus.get(1);
		List<Double> ns = countNs(data_by_class);
		double n1 = ns.get(0);
		double n2 = ns.get(1);
		double n = n1 + n2;
		for (int i = 0; i < n1; i++) {
			for (int j = 0; j < dimension; j++) {
				data_in_c1[i][j] = data_in_c1[i][j] - mu1; 	
			}
		}
		for (int i = 0; i < n2; i++) {
			for (int j = 0; j < dimension; j++) {
				data_in_c2[i][j] = data_in_c2[i][j] - mu2; 	
			}
		}
		Matrix d1 = new Matrix(data_in_c1);
		Matrix d2 = new Matrix(data_in_c2);
		Matrix s1 = d1.transpose().times(d1).times(1/n1);
		Matrix s2 = d2.transpose().times(d2).times(1/n2);
		Matrix s = s1.times(n1/n).plus(s2.times(n2/n));
		return s;
	}
	
	
	public static double calculateW0(Matrix s, List<Double> mus, List<Double> ns) {
		double[][] s_inverse = s.inverse().getArray();
		double mu1 = mus.get(0);
		double mu2 = mus.get(1);
		double n1 = ns.get(0);
		double n2 = ns.get(1);
		double c1_part = 0;
		double c2_part = 0;
		double w0 = 0;
		for (double[] s_row : s_inverse) {
			for (double s_cell : s_row) {
				c1_part += mu1 * s_cell * mu1;
				c2_part += mu2 * s_cell * mu2;
			}
		}
		c1_part = c1_part * (-0.5);
		c2_part = c2_part * (0.5);
		w0 = c1_part + c2_part + Math.log(n1/n2);
		return w0;
	}
	
	public static Matrix calculateW(Matrix s, List<Double> mus) {
		double mu1 = mus.get(0);
		double mu2 = mus.get(1);
		double diff = mu1 - mu2;
		Matrix s_inverse = s.inverse();
		int dimension = s_inverse.getColumnDimension();
		Matrix diffMatrix = new Matrix(dimension, 1, diff);
		return s_inverse.times(diffMatrix);
	}

}
