# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:03:43 2022

@author: neera
"""

#tf.keras.backend.clear_session()
history = model.fit([trainer, trainer2], [train_output, trainer2], batch_size = 8, epochs = 3, shuffle = False, validation_data=([tester, tester2], [test_output, tester2]))