from docarray import Document, DocumentArray

part2_docs = DocumentArray.load_binary('part-1-embeddings.bin', protocol='pickle-array', compress='gzip')
part2_docs.plot_embeddings('embeddings-part1', path='./plot')
