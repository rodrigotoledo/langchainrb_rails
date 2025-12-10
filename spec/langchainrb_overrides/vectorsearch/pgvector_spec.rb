# frozen_string_literal: true

RSpec.describe Langchain::Vectorsearch::Pgvector do
  let(:llm) { double("LLM") }
  let(:model) { double("Model") }
  subject { described_class.new(llm: llm) }

  before do
    subject.model = model
  end

  describe "#similarity_search" do
    it "passes score_threshold to similarity_search_by_vector" do
      allow(llm).to receive(:embed).and_return(double(embedding: [0.1, 0.2]))
      allow(model).to receive(:nearest_neighbors).and_return(double(limit: []))
      allow(model).to receive(:where).and_return([])

      expect(subject).to receive(:similarity_search_by_vector).with(embedding: [0.1, 0.2], k: 4, score_threshold: 0.5)

      subject.similarity_search(query: "test", k: 4, score_threshold: 0.5)
    end
  end

  describe "#similarity_search_by_vector" do
    let(:query) { double("Query") }
    let(:candidates) { double("Candidates") }
    let(:filtered) { [double(id: 1, neighbor_distance: 0.3), double(id: 2, neighbor_distance: 0.4)] }

    before do
      allow(model).to receive(:nearest_neighbors).and_return(query)
    end

    context "without score_threshold" do
      it "returns query.limit(k)" do
        allow(query).to receive(:limit).with(4).and_return(:result)

        result = subject.similarity_search_by_vector(embedding: [0.1, 0.2], k: 4)

        expect(result).to eq(:result)
      end
    end

    context "with score_threshold" do
      it "filters candidates and returns ordered results" do
        allow(query).to receive(:limit).with(9).and_return(candidates) # k + 5 = 9
        allow(candidates).to receive(:select).and_return(filtered)
        allow(filtered).to receive(:first).with(4).and_return(filtered)
        allow(model).to receive(:where).with(id: [1, 2]).and_return(double(order: :ordered_result))

        result = subject.similarity_search_by_vector(embedding: [0.1, 0.2], k: 4, score_threshold: 0.5)

        expect(result).to eq(:ordered_result)
      end
    end
  end

  describe "#ask" do
    it "passes score_threshold to similarity_search and processes results" do
      # Mock embedding
      allow(llm).to receive(:embed).and_return(double(embedding: [0.1, 0.2]))

      # Mock nearest_neighbors and query chain
      query = double("Query")
      allow(model).to receive(:nearest_neighbors).and_return(query)
      allow(query).to receive(:limit).and_return([])

      # Mock search results
      record1 = double("Record1", as_vector: "Vector 1")
      record2 = double("Record2", as_vector: "Vector 2")
      search_results = [record1, record2]

      # Mock similarity_search to return the results
      allow(subject).to receive(:similarity_search).and_return(search_results)

      # Mock logger silence
      logger = double("Logger")
      allow(ActiveRecord::Base).to receive(:logger).and_return(logger)
      allow(logger).to receive(:silence).and_yield

      # Mock generate_rag_prompt
      allow(subject).to receive(:generate_rag_prompt).and_return("Mocked prompt")

      # Mock llm.chat
      chat_response = double("ChatResponse", chat_completion: "Mocked answer")
      allow(llm).to receive(:chat).and_return(chat_response)

      result = subject.ask(question: "question", k: 4, score_threshold: 0.5)

      expect(subject).to have_received(:similarity_search).with(query: "question", k: 4, score_threshold: 0.5)
      expect(subject).to have_received(:generate_rag_prompt).with(question: "question", context: "Vector 1\n---\nVector 2")
      expect(llm).to have_received(:chat).with(messages: [{role: "user", content: "Mocked prompt"}])
      expect(result.chat_completion).to eq("Mocked answer")
    end
  end
end
