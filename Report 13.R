# 0) Prelúdio =======

setwd('/home/heitor/Área de Trabalho/R Projects/Análise Macro/Labs/Lab 13')

library(tidyverse)
library(tidymodels)
library(tensorflow)
library(plotly)
library(GGally) # para os gráficos de correlação
library(ISLR)   # para a base de dados usada

dds <- as.data.frame(Auto) %>% 
	mutate(made_in = 
		   	case_when(Auto$origin==1 ~ 'america',
		   			  Auto$origin==2 ~ 'europe',
		   			  Auto$origin==3 ~ 'japan')) %>%
	mutate(made_in = as.factor(made_in)) %>% 
	select(-origin) %>% 
	as_tibble()

dds %>% str()

# 1) Estatísticas Descritivas =======

dds %>% summary()

ggcorr(dds %>%
	   	select(!c('year', 'name', 'made_in')),
	   label = T)

ggpairs(dds %>%
			select(!c('year', 'name')),
	columns = 1:6,
	ggplot2::aes(colour=made_in, alpha=.3))  %>% ggplotly()

ggplotly( ggplot(
	dds, aes(x=horsepower, y=mpg))+
	geom_point(aes(color=made_in), alpha=.66)+
	geom_smooth(method = 'auto',
				colour='red',
				size = .8,
				se=FALSE))

# 2) Divisão: Treino & Teste =======

slice_1 <-
	initial_split(dds %>%
				  	select(!c('name', 'year',
				  			  'made_in')))
train_dds   <- training(slice_1)
test_dds    <- testing(slice_1)

# 3) Modelo =======

glm_algorit <- linear_reg( penalty = tune()) %>% 
						  # mixture = 1) %>% 
	set_mode('regression') %>% 
	set_engine("keras")

glm_algorit %>% translate()

# 3) Fórmula =======
# Error: spark objects can only be used with the formula interface to `fit()` with a spark data object.
recipe_glm <- recipe(mpg ~ horsepower,
					 data=train_dds) %>%
	step_poly( horsepower,
			   role = "predictor",
			   trained = FALSE,
			   objects = NULL,
			   degree = tune())

#fit_glm <- glm_algorit %>%
#	parsnip::fit(mpg ~ horsepower + poly(horsepower, degree = tune()),
#				 data=train_dds)
# não dá pra tunar o grau no fit :'(

# 4) Workflow =======

glm_wrkflw <- workflow() %>%
	add_model(glm_algorit) %>%
	add_recipe(recipe_glm)

# 5) Validações =======

## 5.1) Reamostragem Cruzada k-Fold ---

vc_kfold <- vfold_cv(train_dds,
					 v=10,
					 repeats = 2)

## 5.2) Reamostragem Cruzada Leave-One-Out ---
## !!! is not good!

loo_dat <- loo_cv(dds %>%
				 	select(!c('year', 'name', 'made_in')))

lm_alone_loo <- linear_reg() %>% 
	set_engine("lm")

splitfun <- function(mysplit){
	fit_split(mpg~horsepower,
				  model=lm_alone_loo,
				  split=mysplit) %>% 
		collect_predictions()}

map(loo_dat$splits,splitfun)

## 5.3) Reamostragem Bootstrap ---

v_boot <- bootstraps(train_dds,
					 times = 5)

## 5.3) Reamostragem Monte-Carlo ---

vc_mc <- mc_cv(train_dds,
			   prop = 4/5,
			   times=3)

# 6) Ajustes e Treinamentos =======

## limites do tune() ---

grid_stand <-
	glm_wrkflw %>% 
	parameters() %>% 
	update(
		penalty = penalty(range = c(-1, 2),
				trans = scales::pseudo_log_trans()),
		degree = degree(range = c(1, 3))
	) %>% 
	grid_regular(levels = 3)

## afinação dos tune() ---

glm_train_kfold <- glm_wrkflw %>% 
	tune_grid(resamples = vc_kfold,
			  grid      = grid_stand,
			  control   = control_grid(save_pred = T),
			  metrics   = metric_set(rmse, rsq))

#glm_train_loo <- glm_wrkflw %>% 
#	tune_grid(resamples = vc_loo,
#			  grid      = grid_stand,
#			  control   = control_grid(save_pred = T),
#			  metrics   = metric_set(rmse, rsq))

glm_train_boot <- glm_wrkflw %>% 
	tune_grid(resamples = v_boot,
			  grid      = grid_stand,
			  control   = control_grid(save_pred = T),
			  metrics   = metric_set(rmse, rsq))

glm_train_mc <- glm_wrkflw %>% 
	tune_grid(resamples = vc_mc,
			  grid      = grid_stand,
			  control   = control_grid(save_pred = T),
			  metrics   = metric_set(rmse, rsq))

glm_train_kfold
#glm_train_loo
glm_train_boot
glm_train_mc

collect_metrics(glm_train_kfold)
#collect_metrics(glm_train_loo)
collect_metrics(glm_train_boot)
collect_metrics(glm_train_mc)

ggplot2::autoplot(glm_train_kfold) +
	labs(title = 'Parâmetros Ajustáveis Testados no k-Fold ')
ggplot2::autoplot(glm_train_boot)
ggplot2::autoplot(glm_train_mc)

glm_train_kfold %>% show_best(n=1)
glm_train_boot  %>% show_best(n=1)
glm_train_mc  %>% show_best(n=1)

# 7) Seleção do melhor modelo =======

best_tune_1  <- select_best(glm_train_kfold, 'rmse')
best_tune_2  <- select_best(glm_train_mc, 'rmse')

final_algotim <- glm_algorit %>%
	finalize_model(parameters = best_tune %>%
				   	dplyr::select(`penalty`))

final_recip_1 <- recipe_glm %>% 
	finalize_recipe(parameters = best_tune_1 %>%
						dplyr::select(`degree`))
final_recip_2 <- recipe_glm %>% 
	finalize_recipe(parameters = best_tune_2 %>%
						dplyr::select(`degree`))

glm_final_wrkflw_1 <- workflow() %>% 
	add_model(final_algotim) %>% 
	add_recipe(final_recip_1) %>% 
	last_fit(slice_1) 

glm_final_wrkflw_2 <- workflow() %>% 
	add_model(final_algotim) %>% 
	add_recipe(final_recip_2) %>% 
	last_fit(slice_1) 

glm_final_wrkflw_1$.metrics
glm_final_wrkflw_1$.predictions
glm_final_wrkflw_2$.metrics
glm_final_wrkflw_2$.predictions

# --- {
glm_final_wrkflw_3 <- finalize_workflow(glm_wrkflw, best_tune)
glm_final_wrkflw_3 
final_fit <- fit(glm_final_wrkflw_3 , train_dds)
final_fit
fit_nls_on_bootstrap <- function(split) {
	nls(mpg ~ k / wt + b, analysis(split), start = list(k = 1, b = 0))
}

# tidymodels.org/learn/statistics/bootstrap/ ---
#boot_models <-
#	boots %>% 
#	mutate(model = map(splits, fit_nls_on_bootstrap),
#		   coef_info = map(model, tidy))
#boot_coefs <- 
#	boot_models %>% 
#	unnest(coef_info)
#boot_aug <- 
#	boot_models %>% 
#	sample_n(200) %>% 
#	mutate(augmented = map(model, augment)) %>% 
#	unnest(augmented)
#ggplot(boot_aug, aes(wt, mpg)) +
#	geom_line(aes(y = .fitted, group = id), alpha = .2, #col = "blue") +
#	geom_point()
# ---

# --- }

glm_final_wrkflw_1 <- glm_final_wrkflw_1 %>%
	collect_predictions()

glm_final_wrkflw_2 <- glm_final_wrkflw_2 %>%
	collect_predictions()

glm_final_wrkflw_1 %>% 
	select(.row, .pred, mpg) %>% 
	ggplot() +
	aes(x = mpg,
		y = .pred) +
	geom_point() +
	geom_abline(intercept = 0,
				slope     = 1,
				color     ='red',
				size      = .8)

glm_final_wrkflw_2 %>% 
	select(.row, .pred, mpg) %>% 
	ggplot() +
	aes(x = mpg,
		y = .pred) +
	geom_point() +
	geom_abline(intercept = 0,
				slope     = 1,
				color     ='lightseagreen',
				size      = .8)









