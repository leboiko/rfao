/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.classifiers.imbalanced;

import moa.classifiers.MultiClassClassifier;
import moa.options.ClassOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.jar.Attributes;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.TestUtils;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.w3c.dom.Attr;

/**
 *
 * @author luiseduardoboikoferreira
 */



public class Rfao extends AbstractClassifier implements MultiClassClassifier {

    public ClassOption baseLearnerOption = new ClassOption(
            "baseLearner", 'l', "", Classifier.class, "bayes.NaiveBayes");
    public IntOption windowOption = new IntOption("window", 'w', "", 1000, 0,
            Integer.MAX_VALUE);

    public FloatOption ratioOption = new FloatOption("ratio", 'r', "", 0.1, 0.0, 1.0);

    public FloatOption expectedCorrelationOption = new FloatOption(
            "expectedCorrelation", 'c', "", 0.1, 0.0, 1.0);
    // adicionar balancear ou nao
    public MultiChoiceOption balanceOption = new MultiChoiceOption("applyBalance", 'b',
            "", new String[]{"True", "False"}, new String[]{"to balance", "or not"}, 0);

    protected Classifier learner;
    protected Integer observedInstances;
    protected Instances batch;
    protected Instances batchMaj;
    protected Instances batchMin;
    protected ArrayList<Double> classes;
    protected ArrayList<Double> classesCount;
    protected ArrayList<Integer> classesDistr;
    protected Double sMaj;
    protected Double sMin;
    protected ArrayList<Attribute> atributos;
    protected DescriptiveStatistics stats;
    protected Integer numInstanciasGerar;
    protected Instances synthInst;
    protected ArrayList<Attribute> atributosInstancia;
    protected HashMap<Attribute, Double> means;
    protected HashMap<Attribute, Double> stdDevs;
    protected HashMap<Attribute, Double> trends;
    protected ArrayList<Attribute> atributosBinarios;
    protected ArrayList<Attribute> atributosNormais;
    protected ArrayList<Attribute> atributosNaoNormais;
    protected ArrayList<CorrelatedPairs> atributosCorrelacionados;

    protected Double balanceLevel;
    @Override
    public double[] getVotesForInstance(Instance instnc) {
        return learner.getVotesForInstance(instnc);
    }

    @Override
    public void resetLearningImpl() {
        learner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        learner.resetLearning();
        this.batch = new Instances();
        this.batchMaj = new Instances();
        this.batchMin = new Instances();
        this.classes = new ArrayList<>();
        this.classesCount = new ArrayList<>();
        this.classesDistr = new ArrayList<>();
        this.observedInstances = 0; // inicializo com valor balanceado
        this.atributos = new ArrayList<>();
        this.stats = new DescriptiveStatistics();
        this.numInstanciasGerar = 0;
        this.synthInst = null;
        this.atributosInstancia = new ArrayList<>();
        this.means = new HashMap<>();
        this.stdDevs = new HashMap<>();
        this.trends = new HashMap<>();
        this.balanceLevel = 0.0;
        this.atributosBinarios = new ArrayList<>();
        this.atributosNormais = new ArrayList<>();
        this.atributosNaoNormais = new ArrayList<>();
        this.atributosCorrelacionados = new ArrayList<CorrelatedPairs>();
    }

    private enum CorrelationKind {
        Normal,
        NotNormal
    }

    @Override
    public void trainOnInstanceImpl(Instance instnc) {
        if(this.observedInstances == 0){
            this.batch = new Instances(instnc.dataset());
            this.batchMaj = new Instances(instnc.dataset());
            this.batchMin = new Instances(instnc.dataset());
        }

        if (this.observedInstances < this.windowOption.getValue()) {
            // Adiciono as classes
            if (!this.classes.contains(instnc.classValue())) {
                this.classes.add(instnc.classValue());
            }


        } else if ((this.observedInstances % this.windowOption.getValue()) == 0) {

            // Atualizo a distribuicao de classes
            this.classesDistr.clear(); // limpo o array que guarda a districuicao de classes
            this.classes.forEach((j) -> {
                this.classesDistr.add(Collections.frequency(this.classesCount, j));
            });

            this.whoIsMaj();
            for(int i = 0; i < batch.numInstances(); i++){
                this.fillBags(batch.get(i));
                this.learner.trainOnInstance(batch.get(i));
            }


            instantiateSynth();

            if ("True".equals(this.balanceOption.getChosenLabel())) {

                this.numInstanciasGerar = this.calcularNumInstanciasGerar();
                this.getStatistics(instnc);
                this.correlationTest(this.atributosNormais, CorrelationKind.Normal);
                this.correlationTest(this.atributosNaoNormais, CorrelationKind.NotNormal);
                this.generateSynthInstances();

                for (int h = 0; h < this.synthInst.numInstances(); h++) {
                    this.learner.trainOnInstance(this.synthInst.get(h));
                }
            }
        } else {
            this.learner.trainOnInstance(instnc);
        }


        if (this.windowOption.getValue() <= this.observedInstances) {
            Instance toRemove = this.batch.get(0);
            this.batch.delete(0);

            if (toRemove.classValue() == this.sMin) {
                this.batchMin.delete(0);
            } else {
                this.batchMaj.delete(0);
            }
        }

        this.batch.add(instnc);
        this.observedInstances++;
        this.classesCount.add(instnc.classValue());
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder sb, int i) {
        sb.append("Teste de texto de coisa");
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    private double[] getArrayOfValues(Attribute attr) {
        double[] arrayValores = new double[this.batchMin.numInstances()];
        // Itero sobre todos os valores correspondentes ao atributo
        int indexOfAtt = this.batchMin.get(0).indexOfAttribute(attr);
        for(int iteRow = 0; iteRow < this.batchMin.numInstances(); iteRow++){
            double value = this.batchMin.get(iteRow).value(indexOfAtt);
            arrayValores[iteRow] = value;
        }
        return arrayValores;
    }


    private void correlationTest(ArrayList<Attribute> arrayAttributes, CorrelationKind kind) {
        for (int i = 0; i < arrayAttributes.size(); i++) {
            double [] pivot = this.getArrayOfValues(arrayAttributes.get(i));
            for (int j = i + 1; j< arrayAttributes.size(); j++) {
                double [] toCompare = this.getArrayOfValues(arrayAttributes.get(j));
                double corr = 0.0;
                if (kind == CorrelationKind.Normal) {
                    corr = new PearsonsCorrelation().correlation(pivot, toCompare);
                } else if (kind == CorrelationKind.NotNormal) {
                    corr = new SpearmansCorrelation().correlation(pivot, toCompare);
                }
                if (corr >= this.expectedCorrelationOption.getValue()) {
                    this.atributosCorrelacionados.add(new CorrelatedPairs(arrayAttributes.get(i),
                            arrayAttributes.get(j)));
                }
            }
        }
    }

    

    private void instantiateSynth() {
        this.synthInst = new Instances(batch);
    }

    private void fillBags(Instance instnc) {
        if (instnc.classValue() == this.sMin) {
            this.batchMin.add(instnc);
        } else {
            this.batchMaj.add(instnc);
        }

    }

    private void classBalanceInBatch() {
        if (this.sMin == this.classes.get(0)) {
            this.balanceLevel = (double) (this.classesDistr.get(0) / this.classesDistr.get(1));
        } else {
            this.balanceLevel = (double) (this.classesDistr.get(1) / this.classesDistr.get(0));
        }
        System.out.println(this.balanceLevel);
    }

    private void whoIsMaj() {
        if (this.classes.size() > 1) {
            if (this.classesDistr.get(0) > this.classesDistr.get(1)) {
                this.sMaj = this.classes.get(0);
                this.sMin = this.classes.get(1);
            } else {
                this.sMaj = this.classes.get(1);
                this.sMin = this.classes.get(0);
            }
        }
    }

    private void getStatistics(Instance instance) {
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            this.getBasicInfo(instance.attribute(i));
        }
    }

    private void getBasicInfo(Attribute atr) {

        double[] arrayOfValues = this.getArrayOfValues(atr);
        for (int i = 0; i < arrayOfValues.length; i++) {
            this.stats.addValue(arrayOfValues[i]);
        }

        // informacoes estatisticas
        double mean = this.stats.getMean();
        double std = this.stats.getStandardDeviation();
        double trend = this.stats.getPercentile(50);
        // gero uma normal para comparar
        final NormalDistribution unitNormal = new NormalDistribution(mean,
                std);
        double pvalue = TestUtils.kolmogorovSmirnovStatistic(unitNormal,
                arrayOfValues);
        if (pvalue <= 0.05) {
            if (!this.atributosNormais.contains(atr)) { this.atributosNormais.add(atr); }
        } else {
            if (!this.atributosNaoNormais.contains(atr)) { this.atributosNaoNormais.add(atr); }
        }

        //stores mean and stddev
        this.means.put(atr, mean);
        this.stdDevs.put(atr, std);
        this.trends.put(atr, trend);
        this.stats.clear();
    }



    private double generateSynthValuesByMean(/*Attribute atributo, */Double mean, Double std) {
        double rangeMin = (mean - std);
        double rangeMax = (mean + std);
        double value = rangeMin + (rangeMax - rangeMin) * this.classifierRandom.nextDouble();
        return value;
    }


    private void generateSynthInstances() {
        this.synthInst.delete();
        double v = 0.0;
        for (int k = 0; k < this.numInstanciasGerar; k++) {
            Instance synt = new DenseInstance(batch.numAttributes());
            synt.setDataset(synthInst);

            for (int l = 0; l < this.batchMin.numAttributes(); l++) {
                if (l != this.batch.get(0).classIndex()) {
                    Attribute att = this.batch.get(0).attribute(l);

                    if (att.isNumeric()) {
                        v = generateSynthValuesByMean(this.means.get(this.batch.get(0).attribute(l)),
                                this.stdDevs.get(this.batch.get(0).attribute(l)));
                    } else if (att.isNominal()) {
                        if (att.numValues() == 2) {
                            v = this.trends.get(att);
                        } else {
                            // TODO fazer gerar via distribuicao de probabilidade
                            v = this.trends.get(att);
                        }
                    }

                    synt.setValue(l, v);
                }

            }

            // sets the class
            synt.setClassValue(this.sMin);
            this.synthInst.add(synt);
        }
        System.out.printf("Batch synth:    %d\n", this.synthInst.size());


    }


    private int calcularNumInstanciasGerar() {
        int quantasInstanciasDeveriaTer = (int) (this.batchMaj.size() * this.ratioOption.getValue());
        return quantasInstanciasDeveriaTer - this.batchMin.size();
    }

    private class CorrelatedPairs{
        Attribute a;
        Attribute b;
        CorrelatedPairs(Attribute a, Attribute b){
            this.a = a;
            this.b = b;
        }

    }

}

