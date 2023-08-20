library(tidyr)
library(ggplot2)

setwd('~/csboy/lm-ditransitive/analysis')
df <- read.csv('ngram.csv', sep = '\t')

`%nin%` <- Negate(`%in%`)

df[df$n %in% c('n = 1', 'n = 5', 'n = 6', 'n = 7'), ] %>% ggplot(aes(x = factor(theme, levels = c('short', 'long')),
                  y = pref,
                  fill = factor(recipient, levels = c('short', 'long')))) +
  geom_bar(stat = 'identity', position = 'dodge') +
  labs(x = 'Theme Length', fill = 'Recipient Length', y = 'PO Preference (bits)') +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.4, position=position_dodge(.9), color = 'black', linewidth = 0.25) +
  theme(legend.position="bottom") +
  facet_grid(factor(n) ~ factor(k), scales = "free_x", space = "free_x", switch = "y")

ggsave('ngram_prefs.pdf', width = 7, height = 10, unit = 'in')
ggsave('ngram_prefs.png', width = 7, height = 10, unit = 'in')


df1 <- df[df$n == 'n = 5', ]
df2 <- df1[df1$k == 'k = 2',]

