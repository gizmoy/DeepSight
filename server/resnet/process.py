import os
import tensorflow as tf

from random import shuffle



# Configuration
tf.app.flags.DEFINE_string('synth_root', 'C:/Users/Mike/Documents/mnt/ramdisk/max/90kDICT32px', """Path to the root of a SynthText annotations""")
tf.app.flags.DEFINE_string('chunks_dir', './chunks2', """Directory where to write chunks of SynthText annotations""")
tf.app.flags.DEFINE_integer('chunk_size', 1000, """How many classes falls into one chunk""")

FLAGS = tf.app.flags.FLAGS


def chunk():
    print('[Configuration]')
    print('\tSynthText root: %s' % FLAGS.synth_root)
    print('\tChunk directory: %s' % FLAGS.chunks_dir)
    print('\tChunk size: %s' % FLAGS.chunk_size)


    # Load lexicon and count number of words
    with open(os.path.join(FLAGS.synth_root, 'lexicon.txt')) as lex_file:
        words = lex_file.read().splitlines()
        format_str = 'lexicon has %d words'
        print(format_str % len(words))
        
    # Create indecies list, shuffle and chuck it
    indecies = list(range(len(words)))
    shuffle(indecies)
    chsize = FLAGS.chunk_size
    indecies_chunks = [indecies[i:i + chsize] for i in range(0, len(indecies), chsize)]

    # Iterate over indecies chunks and profiles
    for j, chunk in enumerate(indecies_chunks):
        for profile in ['train', 'val', 'test']:         

            # Open Synth profile annotations file
            synth_filename = 'annotation_' + profile + '.txt'
            synth_path = os.path.join(FLAGS.synth_root, synth_filename)

            with open(synth_path) as synth_file:
                # Read lines  
                lines = synth_file.read().splitlines()

                # Print number of examples at first iteration
                if j == 0:
                    format_str = '(%s)   has %d examples'
                    print(format_str % (profile, len(lines)))

                # Verbose
                format_str = '(%s_%d)   loading chunk'
                print(format_str % (profile, j))
        
                # Create file
                out_filename = profile + '_' + str(j) + '.txt'
                out_path = os.path.join(FLAGS.chunks_dir, out_filename)
                if os.path.exists(out_path):
                    os.remove(out_path)

                # Filter profile image annotations
                filtered_lines = [l for l in lines if int(l.split()[1]) in chunk]

                # Open file to write
                with open(out_path, 'a') as out_file:
                    num_correct = 0
                    incorrect = []
                    for i, line in enumerate(filtered_lines):
                        out_file.write(line + '\n')

                        # Print some info periodically
                        if i % 10000 == 0 or i == len(filtered_lines) - 1:
                            format_str = '\t %d/%d completed'
                            print(format_str % (i, len(filtered_lines)))


def main(argv=None):
  chunk()


if __name__ == '__main__':
  tf.app.run()