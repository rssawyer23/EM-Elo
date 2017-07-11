## SCRIPT FOR EXPLORATORY DATA ANALYSIS
data = read.csv("NBAPointSpreadsAugmented.csv")
h = data[,"Home.Points"]
mu_h = mean(h);mu_h
sd_h = sd(h);sd_h
a = data[,"Away.Points"]
mu_a = mean(a);mu_a
sd_a = sd(a);sd_a
margin = h - a
mu_margin = mean(margin)
sd_margin = sd(margin)
cor.test(a,h)
t.test(a,h)
var.test(a,h)
var.test(h,a)

## Test of Equal Variance
F_stat = sd_a^2 / sd_h^2
df_ah = dim(data)[1] - 1
p_value = 1 - pf(F_stat, df_ah, df_ah);p_value

## Visual Representations
hist(a, breaks=100, main="Away Total Distribution", xlab="Away Points")
# Plot density using mean(a) and sd(a)
hist(h, breaks=100, main="Home Total Distribution", ylab="Home Points")
# Plot density using mean(h) and sd(h)
mu_margin = mean(margin)
sd_margin = mean(margin)
hist(margin, breaks=100, main="Home Margin Histogram", xlab="Home Margin")
# Plot a normal curve on this useing mu_margin and sd_margin
# Plot a Gaussian mixture model curve here

## Visualizing Game Scores
df = data.frame(h, a)
with(df, plot(h, a, col="#00000013", pch=19, xlim=c(60,150), ylim=c(60,150),
	main="Game Score Plot", xlab="Home Points", ylab="Away Points"))
x=c(1,2)
y=c(1,2)
abline(lm(y~x),lty=3)
abline(lm(a~h),lty=3)
# Plot the Bivariate normal density using the avg and std of h and a
cov_ah = cor(a,h)*sd(a)*sd(h)
mu_v = c(mu_h, mu_a)
cov_m = cbind(c(var(h),cov_ah),c(cov_ah, var(a)))

