package com.Weka.first;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.unsupervised.attribute.NumericToNominal;




public class WekaTest {
	 public static void main(String[] args) throws Exception {  
	        Classifier m_classifier = new J48();  
	        // 训练语料文件  
	        File inputFile = new File("C:/Program Files/Weka-3-6/data/segment-challenge.arff");  
	        ArffLoader atf = new ArffLoader();  
	        atf.setFile(inputFile);  
	        // 读入训练文件  
	        Instances instancesTrain = atf.getDataSet();  
	        instancesTrain.setClassIndex(instancesTrain.numAttributes()-1);  
	        // 训练  
//	        NumericToNominal numToNom=new NumericToNominal();
//			try {
//				numToNom.setAttributeIndices("" + (instancesTrain.classIndex() + 1));
//				numToNom.setInputFormat(instancesTrain);
//				instancesTrain=numToNom.useFilter(instancesTrain, numToNom);
//			} catch (Exception e2) {
//				// TODO Auto-generated catch block
//				e2.printStackTrace();
//			}
	        m_classifier.buildClassifier(instancesTrain);  
	  
	        // 测试语料文件  
	        inputFile = new File("C:/Program Files/Weka-3-6/data/segment-test.arff");  
	        atf.setFile(inputFile);  
	        // 读入测试文件  
	        Instances instancesTest = atf.getDataSet();  
	        // 设置分类属性所在行号（第一行为0号），instancesTest.numAttributes()可以取得属性总数  
	        instancesTest.setClassIndex(instancesTest.numAttributes()-1);  
	  
	        // 测试语料实例数  
	        double sum = instancesTest.numInstances();  
	        double right = 0.0f;  
	        // 测试分类结果  
	        for (int i = 0; i < sum; i++) {  
	            // 如果预测值和答案值相等（测试语料中的分类列提供的须为正确答案，结果才有意义）  
	            if (m_classifier.classifyInstance(instancesTest.instance(i))==( instancesTest.instance(i).classValue())) {  
	                // 正确值加1  
	                right++;  
	            }  
	        //	System.out.println(m_classifier.classifyInstance(instancesTest.instance(i))+"  "+instancesTest.instance(i).classValue());
	        }  
	        System.out.println("J48 classification precision:" + (right / sum));  
	    }  
	
}
