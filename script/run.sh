# https://stackoverflow.com/questions/59371228/albert-base-weights-from-ckpt-not-loaded-properly-when-calling-with-bert-for-t
# https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96

python tensorflow_rename_variables.py --checkpoint_dir=../proprietary/korbert/002_bert_morp_tensorflow/model.ckpt --replace_from="module/bert/" --replace_to="bert/" --add_prefix="" --dry_run




# dhruvsakalley commented on 7 Mar
# Alright after some digging around this what I found, would love some confirmation on it:
#
# Based on the response provided by a member from the BERT team, the fine-tuned model is 3 times larger than the distributed checkpoint due to the inclusion of Adam momentum and variance variables for each weight variable. Both variables are needed to be able to pause and resume training. In order words, if you intend to serve your fine-tuned model without any further training, you can remove both variables and and the size will be more or less similar to the distributed model.
# https://towardsdatascience.com/3-ways-to-optimize-and-export-bert-model-for-online-serving-8f49d774a501
# So it seems I can ignore those for the init checkpoint model, it seems like the code handles those variables being absent, would love some confirmation from Bert folks though.
#  @manueltonneau
#
# manueltonneau commented on 24 Mar •
# edited
# These variables are indeed the momentum and variance from the Adam optimizer. These variables are built during training and are needed to pause and resume training. The distributed checkpoints (the ones you can download on the repos) don't contain these Adam variables and this is why they are three times smaller. See answer of Jacob Devlin here.
#
# I personally faced issues trying to initialize from distributed checkpoints because of the absence of these Adam variables, happy to know more if you managed to do it! :)



# saver.save(sess, variables_file_path, write_meta_graph=False, write_state=True)  # write state is true by default.