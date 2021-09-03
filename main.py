from open_game import Game

game = Game()
a = game.agent


game.run(agent=a, num_games=10, num_rounds=25, batch_size=5,  training=True)
story1 = game.story.copy()


story1.to_csv('story3.csv')



