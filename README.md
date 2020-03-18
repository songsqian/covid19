This repository includes R program for estimating the prevalence of COVID-19 pandemic virus in the US. Data used in this repository are collected by the [COVID-19 Tracking Project](https://github.com/COVID19Tracking/covid-tracking-data). The R program is written in a Python notebook.  It can be executed from Google Colab.

The basic statistical approach is documented by [Qian et al., 2020](https://urldefense.com/v3/__https://authors.elsevier.com/sd/article/S2405844020304163__;!!LoBwcKfm!wzx5UXnrlWrx7W0yAaMR_h11cYYkUVmCovr3pExJN50C0BukosQMlBqefUw5hxaA--3d$), along with the [computational details](github.com/songsqian/imperfect).  The basic concept is that the test result is uncertain because of the inevitable false positive and false negative outcomes.  To properly interpret the test result, either by a patient or by the state health authority, we must translate the result into the relevant quantity. Because of the inevitable false positive, a positive result cannot be equated to the presence of the virus; likewise, the likelihood of false negatives makes a negative result less assuring.  As a result, a positive (negative) result should be interpreted in terms of the probability of infection (non-infection), specifically, the conditional probability of infection given a positive result. Let $+$ ($-$) represent a positive (negative) result, $v$ represent the presence of the virus, and $a$ represent the absence of the virus.  When observing $+$ for an individual patient, we want to know $\Pr(v|+)$, which reads "the conditional probability of $v$ given $+$."  This conditional probability is part of the Bayes Theorem:
$$
\Pr(v|+) = \frac{\Pr(v)\Pr(+|v)}{\Pr(v)\Pr(+|v)+\Pr(a)\Pr(+|a)}
$$
In other words, to learn about the meaning of a positive result, we need to know three more quantities: $\Pr(+|v)$ (probability of a positive result when the virus is present, or 1 minus the probability of a false negative), $\Pr(+|a)$ (the probability of a false positive), and $\Pr(v)$ the prevalence of the virus infection in the population.  Interpretation of $\Pr(v)$ depends on the definition of the population \citep{Qian.etal2020}.

Probabilities of false positive and false negative are features of the test and the prevalence of the infection is what we, as a society, want to learn from the repeated testing.  At this point, we have no definite knowledge on these three quantities.  Therefore, from the perspective of government health authorities, we want to use test results to learn about these quantities so that individual patients can better understand the meaning of the test results.

Following the notation of \citet{Qian.etal2020}, let $\theta$ be the prevalence ($\theta=\Pr(v)$), $f_p$ the false positive rate, and $f_n$ false negative rate, the statistical model for updating the probability distribution of $\theta$ is the continuous variable version of the Bayes theorem.
$$
\pi(\theta|y, n) = \frac{\pi(\theta)L(\theta|y, n)}{\int\pi(\theta)L(\theta|y, n)d\theta}
$$
where $y$ and $n$ are numbers of positive and total tests and $L(\theta|y, n)$ is the likelihood function (representing the probability of observing $y$ positives out of $n$ tests).  The likelihood is derived based on the binomial distribution assumption of $y$, and it is a function of $\theta$, $f_p$, and $f_n$ \citep{Qian.etal2020}:
$$
\begin{array}{rcl}
p_+ & = & \theta(1-f_n)+f_p(1-\theta)\\
L(\theta, f_p, f_n|y, n) &\propto& p_+^{y}(1-p_+)^{n-y}
\end{array}
$$
where $p_+$ is the probability of observing a positive result. Because we don't know $f_p$ and $f_n$, we use the Bayes theorem to update them as well:
$$
\pi(\theta, f_p, f_n|y, n) = \frac{\pi(\theta)\pi(f_p)\pi(f_n)L(\theta, f_p, f_n|y, n)}{\int_\theta\int_{f_p}\int_{f_n}\pi(\theta)\pi(f_p)\pi(f_n)L(\theta,f_p,f_n|y, n)d\theta df_p df_n}
$$
As the three quantities of interest ($\theta, f_p, f_n$) are probabilities, we use the beta distribution as their priors.  

### Prior specification

As the COVID-19 virus is a new virus and only a relatively small number of tests are done in the US (only for people with specific symptoms), we don't have a basis for specifying a more informative prior.  As a result, we used the commonly used non-informative beta prior ($beta(1,1)$).  For the qPCR test used for detecting the COVID-19 virus, we haven't seen studies to quantify $f_p$ and $f_n$. However, the basic principle of the test is well known.  We can use reported $f_p$ and $f_n$ for similar types of tests to develop the priors. We estimate state-level prevalence under three scenarios. 

- Estimates assuming a highly accurate test
  Assuming the test has a false positive probability of 1% and a false negative probability of 1% and both are stable ($f_p\sim beta(1,99)$ and $f_n\sim beta(1,99)$, with a 95% credible interval of  0.00026 0.036).
- Estimates based on similar tests for other corona-virus (SARS, MERS, H1N1)
  Existing studies on tests of similar viruses have a range of false positive and false negative rates.  Using a study of the MERS and SARS virus tests (REF), we constructed a prior distribution for false positive to be $beta(3,23)$ (95% credible interval of 2.5%-26%) and false negative $beta(2,22)$ (1%-22%)
- Estimates assuming the test is as unreliable as the rapid influenza diagnostic test (RIDT) (REF)
  This test has a high false negative probability and we use $beta(16,24)$ (0.26-0.55).  Probability of false positive is relatively low and we used $beta(4,45)$ (0.023, 0.17).

### A hierarchical formulation

We have data from nearly all 50 states. We assume that $f_p$ and $f_n$ are the same for all states because they use the same test. However, the prevalence can vary by region.  But we have no information to separate one state from another other than the testing data.  As a result, we assume that the prevalence for each state $\theta_j$ are exchangeable and impose a common prior.  Expressing the model hierarchically, we have the following mode.

1. At the observational level, data from each state (numbers of positive and negative) are modeled by the binomial distribution

$$
y_j \sim Bin(p_j, n_j)
$$

where $j$ represents the $j$th state, $y_j$ and $n_j$ are the observed number of positive and total number of tests.  The probability of observing a positive result ($p_j$) is a function of $\theta_j$, $f_p$, and $f_n$:
$$
p_j = \theta_j(1-f_n)+(1-\theta)f_p
$$

2. To connect all states together, we use a common prior for state-level prevalence $\theta_j$.   

$$
\mathrm{logit}(\theta_j)  \sim  N(\mu_0, \sigma_0^2)
$$

3. Prior distributions of other parameters

$$
\begin{array}{rcl}
f_n &\sim& beta(\alpha_n, \beta_n)\\
f_p &\sim& beta(\alpha_p, \beta_p)
\end{array}
$$

The hyper-parameter $\mu_0$ is the national average of prevalence (in logit scale) and $\sigma_0^2$ is among state variance of the logit transformed state-specific prevalence. # covid19
