package moa.classifiers.teste;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.Measurement;
import moa.options.ClassOption;
import weka.classifiers.AbstractClassifier;

public class teste extends moa.classifiers.AbstractClassifier implements MultiClassClassifier {
    public ClassOption baseLearnerOption = new ClassOption(
            "baseLearner", 'l', "", Classifier.class, "bayes.NaiveBayes");
    public IntOption windowOption = new IntOption("window", 'w', "", 1000, 0,
            Integer.MAX_VALUE);
    public IntOption nHats = new IntOption("nHats", 'n', "", 1000, 0,
            Integer.MAX_VALUE);

    private Classifier learner;
    private Instances window; // criar uma janela deslizante
    private Integer observedInstances; // criar uma variavel para guardar o contador de instancias observadas

    @Override
    public void trainOnInstanceImpl(Instance instnc) {

        if (this.observedInstances == 0) {
            this.window = new Instances(instnc.dataset());
        } else if ((this.observedInstances % this.windowOption.getValue()) == 0) {
            // aqui acontece o codigo a cada janela
            for (int i = 0; i < this.window.size(); i++) {
                this.learner.trainOnInstance(this.window.get(i));
            }
        }

        // atualizar a janela deslizante
        if (this.window.size() >= this.windowOption.getValue()) {
            this.window.delete(0);
        }
        this.window.add(instnc);

        this.observedInstances ++;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return new double[0];
    }

    @Override
    public void resetLearningImpl() {
        this.window = new Instances();
        this.observedInstances = 0;
        this.learner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        this.learner.resetLearning();
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
        return false;
    }
}
