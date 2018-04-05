import os
import tensorflow as tf

from random import shuffle
from shutil import copyfile



# Configuration
tf.app.flags.DEFINE_string('synth_root', 'C:/Users/Mike/Documents/mnt/ramdisk/max/90kDICT32px', """Path to the root of a SynthText annotations""")
tf.app.flags.DEFINE_string('chunks_dir', './chunks_2000', """Directory where to write chunks of SynthText annotations""")
tf.app.flags.DEFINE_integer('chunk_size', 2000, """How many classes falls into one chunk""")
tf.app.flags.DEFINE_string('shuffle_name', 'shuffled.txt', """Path to shuffled ids file""")

FLAGS = tf.app.flags.FLAGS


def chunk():
    print('[Configuration]')
    print('\tSynthText root: %s' % FLAGS.synth_root)
    print('\tChunk directory: %s' % FLAGS.chunks_dir)
    print('\tChunk size: %s' % FLAGS.chunk_size)
    print('\tName of file with shuffled ids: %s' % FLAGS.shuffle_name)

    # Create chunk directory if does not exist
    if not os.path.exists(FLAGS.chunks_dir):
        os.makedirs(FLAGS.chunks_dir)

    # Load lexicon and count number of words
    with open(os.path.join(FLAGS.synth_root, 'lexicon.txt')) as lex_file:
        words = lex_file.read().splitlines()
        words_dict = dict([(id, word) for id, word in enumerate(words)])
        format_str = 'lexicon has %d words'
        print(format_str % len(words))
        
    # Create ids list, shuffle and save it if necessary
    ids = list(range(len(words)))
    shuffle(ids)
    shuffle_path = os.path.join(FLAGS.chunks_dir, FLAGS.shuffle_name)
    if os.path.exists(shuffle_path):
        os.remove(shuffle_path)
    with open(shuffle_path, 'a') as shuffle_file:
        for i, id in enumerate(ids):
            shuffle_file.write(str(i) + ' ' + str(id) + ' ' + words_dict[id] + '\n')
    ids_dict = dict([(str(id), i) for i, id in enumerate(ids)])

    # Chunk ids
    ids_chunks = [ids[i:i + FLAGS.chunk_size] for i in range(0, len(ids), FLAGS.chunk_size)]

    # Iterate over ids' chunks and profiles
    for j, chunk in enumerate(ids_chunks):
        for profile in ['train', 'val', 'test']:  

            # Verbose
            format_str = '(%s_%d) loading chunk'
            print(format_str % (profile, j))       

            # Open SynthText profile annotation file
            synth_filename = 'annotation_' + profile + '.txt'
            synth_path = os.path.join(FLAGS.synth_root, synth_filename)

            # Read lines
            with open(synth_path) as synth_file:
                lines = synth_file.read().splitlines()

            # Print number of examples at first iteration
            if j == 0:
                format_str = '(%s) has %d examples'
                print(format_str % (profile, len(lines)))

            # Filter profile image annotations
            filtered_lines = [l for l in lines if int(l.split()[1]) in chunk]

            # Delete output file if does exist
            out_filename = profile + '_' + str(j) + '.txt'
            out_path = os.path.join(FLAGS.chunks_dir, out_filename)
            if os.path.exists(out_path):
                os.remove(out_path)

            # Copy content of previous profile file if iteration greater than 0
            if j > 0:
                prev_out_filename = profile + '_' + str(j-1) + '.txt'
                prev_out_path = os.path.join(FLAGS.chunks_dir, prev_out_filename)
                copyfile(prev_out_path, out_path)

            # Open output file
            with open(out_path, 'a') as out_file:
                for i, line in enumerate(filtered_lines):
                    path, id = line.split()
                    out_file.write(path + ' ' + str(ids_dict[id]) + '\n')
                        
                    # Print some info periodically
                    if i % 10000 == 0 or i == len(filtered_lines) - 1:
                        format_str = '\t %d/%d completed'
                        print(format_str % (i, len(filtered_lines)))


def main(argv=None):
  chunk()


if __name__ == '__main__':
  tf.app.run()