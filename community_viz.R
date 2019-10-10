library(tidyverse)
library(ggforce)
library(ggraph)
library(tidygraph)
library(igraph)

nodes <- read_csv('throughput/nodes.csv')
nodes <- nodes %>%
	 arrange(group)
edges <- read_csv('throughput/edges.csv')

node_dummy <- tibble(
	index = -(1:8),
	group = rep(0:3,2),
	name = NA,
	job = NA,
	node_id = -(1:8),
	other = NA
)

nodes <- nodes %>% 
	rbind(node_dummy) %>% 
	mutate(dummy = ifelse(index < 0, 0, 1),
				 highlight = ifelse(job %in% c('CEO', 'President'), T, F),
				 highlight = ifelse(is.na(highlight), F, highlight),
				 job = factor(job, 
				 						 levels = c(
				 						 	'CEO', 
				 						 	'President', 
				 						 	'Vice President', 
				 						 	'Director', 
				 						 	'Manager',
				 						 	'Trader',
				 						 	'Employee',
				 						 	'In House Lawyer'
				 						 ))) %>% 
	arrange(group, job)

G <- graph_from_data_frame(edges, vertices = nodes)

G_ <- G %>%
	as_tbl_graph() %>%
	activate(edges) %>%
	filter(weight >= 0) %>%
	activate(nodes) %>% 
	mutate(job = factor(job, 
											levels = c(
												'CEO', 
												'President', 
												'Vice President', 
												'Director', 
												'Manager',
												'Trader',
												'Employee',
												'In House Lawyer'
											)),
				 name = ifelse(job %in% c('CEO', 'President'), name, NA)) 

ggraph(G_, layout = 'linear', circular = TRUE) +
	geom_edge_arc(aes(alpha = weight)) + 
	geom_node_point(aes(fill = job,  
											alpha = dummy, 
											stroke = highlight), size = 3, pch = 21) + 
	guides(edge_alpha = F, shape = F, alpha = F) + 
	theme_void() +
	scale_alpha_continuous(range = c(0,1)) + 
	scale_edge_alpha_continuous(range = c(0.00,1)) + 
	guides(fill = guide_legend(title = 'Title')) 
# + 
# 	geom_node_label(aes(label = name), repel = T, size = 2, force = 1)

ggsave('fig/community.pdf', width = 5.5, height = 4)
