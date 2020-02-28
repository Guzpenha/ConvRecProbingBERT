library(ggplot2)

df = read.csv("/Users/gustavopenha/personal/recsys20/vis/conv_data_statistics.csv")

ggplot(df, aes(x=subreddit, y=words_context)) + 
  geom_boxplot(outlier.shape = NA) + coord_cartesian(ylim=c(0,200))+
  ylab("Number of words in contexts")

ggplot(df, aes(x=subreddit, y=words_response)) + 
  geom_boxplot(outlier.shape = NA) + coord_cartesian(ylim=c(0,200)) +
  ylab("Number of words in responses")

ggplot(df, aes(x=relevance_score, color=subreddit)) + stat_ecdf()+
  coord_cartesian(xlim=c(0,15)) + 
  theme(legend.position = c(0.8, 0.2)) + ylab("% of responses") + xlab("relevance_score <=")

ggplot(df, aes(x=turns, y=log(..count..), fill=subreddit)) +
  geom_histogram(position="dodge",bins=6)+
  theme(legend.position = c(0.7, 0.8))
