import React from 'react';
import ReactTable from 'react-table'
import 'react-table/react-table.css';
import '../scss/highlighter.css';
import '../scss/react-table.css';
import styled from 'styled-components';
import Highlighter from 'react-highlight-words';
import { shouldUpdate } from '../utils';

const initialState = {
    currentPage: 0,
    expandedPerPage: {},
    activeQuestions: {},
    selectedAnswers: {}
};

const MAX_LENGTH_PER_PAGE = 100
const allExpanded = []
for (let i=0; i < MAX_LENGTH_PER_PAGE; i++) {
     allExpanded.push(true);
}
const updateSignals = ['dataset', 'expandAll', 'currentPage', 'expandedPerPage', 
                        'activeQuestions', 'selectedAnswers', 'filteredAnswerTypes']
class ExplorerTable extends React.Component {
    constructor(props) {
        super(props);
        renderPassageCell = renderPassageCell.bind(this);
        renderQuestionCell = renderQuestionCell.bind(this);
        renderAnswerCell = renderAnswerCell.bind(this);
        selectedAnswerChange = selectedAnswerChange.bind(this);
        this.clearSelectedAnswers = this.clearSelectedAnswers.bind(this)

        this.filteredDatasets = {}
        this.hasValidatedAnswers = false;
        this.state = initialState;
    }

    componentDidMount() {
        this.props.sendClearSelectedAnswersFunc(this.clearSelectedAnswers)
    }

    shouldComponentUpdate(nextProps, nextState) {
        if (this.props.dataset !== nextProps.dataset) {
            // reset all when the dataset is switched
            this.filteredDatasets = {};
            this.setState(initialState);
        }
        if (this.props.expandAll !== nextProps.expandAll ||
            this.state.expandAll !== nextState.expandAll) {
            if (nextProps.expandAll) {
                this.setState({ expandAll: true })
            } else if (nextState.expandAll) {
                // expandAll was changed from true to false, so reset
                this.setState({
                    expandedPerPage: initialState['expandedPerPage'],
                    expandAll: false
                });
            }
        }

        return shouldUpdate(updateSignals, this.props, this.state, nextProps, nextState)
    }

    handlePageChanged(newPage) {
        this.setState({ currentPage: newPage });
    }

    handleRowExpanded(newExpanded, index, event) {
        if (!this.state.expandAll) {
            let expandedInPage = this.state.expandedPerPage[this.state.currentPage]
            if (!expandedInPage) {
                expandedInPage = {}
            }
            this.setState({
                expandedPerPage: {
                    ...this.state.expandedPerPage,
                    [this.state.currentPage]: {
                        ...this.state.expandedPerPage[this.state.currentPage],
                        [index[0]]: !expandedInPage[index[0]],
                    }
                }
            });
        }
    }

    clearSelectedAnswers() {
        this.setState({
            activeQuestions: initialState.activeQuestions,
            selectedAnswers: initialState.selectedAnswers
        });
    }

    getExpanded() {
        if (this.state.expandAll) {
            return allExpanded;
        }
        return this.state.expandedPerPage[this.state.currentPage]
    }

    render() {
        const filteredAnswerTypes = [...this.props.filteredAnswerTypes].sort();

        let data = this.props.dataset;
        let hasValidatedAnswers = false;
        if (data && filteredAnswerTypes.length > 0) {
            const cacheKey = filteredAnswerTypes.join('|')
            if (!this.filteredDatasets[cacheKey]) {

                let process_row = (accumulator, row, index) => {
                    const qa_pairs = row.qa_pairs;

                    const reduced_qa_pairs = qa_pairs.reduce(process_qa_pair, {
                        passage_id: row.passage_id, 
                        reduced_qa_pairs: []
                    }).reduced_qa_pairs;

                    if (reduced_qa_pairs.length > 0) {
                        const reduced_row = {
                            ...row,
                            qa_pairs: reduced_qa_pairs
                        }
                        accumulator.push(reduced_row);
                    }
                    return accumulator;
                }

                let process_qa_pair = (accumulator, qa_pair, index) => {
                    const passage_id = accumulator.passage_id;

                    if (!hasValidatedAnswers && qa_pair.validated_answers && qa_pair.validated_answers.length > 0) {
                        hasValidatedAnswers = true;
                    }

                    if (filterQAPairByAnswerType(qa_pair, filteredAnswerTypes)) {
                        accumulator.reduced_qa_pairs.push({
                            ...qa_pair,
                            query_index: index,
                            passage_id: passage_id
                        })
                    }
                    return accumulator;
                };

                const filteredData = data.reduce(process_row, [])
                data = filteredData;

                this.filteredDatasets[cacheKey] = data;
                this.hasValidatedAnswers = hasValidatedAnswers;
            } else {
                data = this.filteredDatasets[cacheKey];
                hasValidatedAnswers = this.hasValidatedAnswers;
            }
        } else {
            data = []
        }
       
        const passage_columns = [
            {
                Header: '#',
                id: 'passage_index',
                accessor: 'passage_index',
                width: 50
            },
            {
                Header: 'Passage ID',
                accessor: 'passage_id',
                width: 110
            }, 
            {
                Header: 'Passage',
                accessor: 'passage',
                Cell: renderPassageCell
            }
        ]

        const qa_columns = [
            {
                Header: '#',
                accessor: 'query_index',
                width: 30
            },
            {
                Header: 'Question ID',
                accessor: 'query_id',
                width: 100
            },
            {
                Header: 'Question',
                accessor: 'question',
                Cell: renderQuestionCell
            },
            {
                Header: 'Answer',
                id: 'answer',
                accessor: qa_pair => {
                    const answerField = getAnswerField(qa_pair.answer); 
                    if (answerField) {
                        return getAnswerForDisplay(qa_pair.answer[answerField.key]).toString();
                    }
                    return ''
                },
                Cell: renderAnswerCell,
                width: 150
            },
            {
                Header: 'Answer Type',
                id: 'answerType',
                accessor: qa_pair => {
                    const answerField = getAnswerField(qa_pair.answer);
                    if (answerField) {
                        return answerField.name;
                    }
                    return '';
                },
                width: 110
            },
            {
                Header: 'Additional Answers',
                id: 'additional_answers',
                show: hasValidatedAnswers,
                accessor: qa_pair => {
                    if (!qa_pair.validated_answers || qa_pair.validated_answers.length === 0) {
                        return null;
                    }
                    
                    const answers = [];
                    qa_pair.validated_answers.forEach(answerDict => {
                        const answerField = getAnswerField(answerDict);
                        if (answerField) {
                            answers.push(getAnswerForDisplay(answerDict[answerField.key]));
                        }
                    });
                    return JSON.stringify(answers, null, 2);
                },
                Cell: props => <WrapDiv>{ props.value }</WrapDiv>,
                width: 150
            },
        ]

        return <div>
            <ReactTable className="-striped-passage -highlight-passage"
            data={data} 
            columns={passage_columns}
            minRows={0}
            showPaginationTop={true}
            showPaginationBottom={true}
            pageSizeOptions={[1, 5, 10, 20, 25, 50, 100]}
            onPageChange={newPage => this.handlePageChanged(newPage)}
            expanded={this.getExpanded()}
            onExpandedChange={(newExpanded, index, event) => this.handleRowExpanded(newExpanded, index, event)}
            SubComponent={row => {
                    const qa_pairs = row.original.qa_pairs

                    return (
                        <ReactTable className="-striped-question -highlight-question"
                        data={qa_pairs}
                        columns={qa_columns}
                        minRows={0}
                        defaultPageSize={1000}
                        showPagination={false}
                        getTdProps={(state, rowInfo, column, instance) => {
                            return {
                                onClick: (e, handleOriginal) => {
                                    if (column.id === 'answer') {
                                        selectedAnswerChange(rowInfo, e);
                                    }

                                    if (handleOriginal) {
                                        handleOriginal();
                                    }
                                }
                            }
                        }}
                        />
                    )
                }}
            /></div>
    }
}

const WrapDiv = styled.div`
    white-space: pre-wrap;       /* css-3 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
`;

let renderPassageCell = function(props) {
    const passage_id = props.original.passage_id;
    const selectedAnswers = this.state.selectedAnswers[passage_id]
    const searchWords = selectedAnswers ? selectedAnswers : []
    return <WrapDiv><Highlighter highlightClassName="highlight-gold" searchWords={searchWords} textToHighlight={props.value} /></WrapDiv>
}
let renderQuestionCell = function(props) {
    let searchWords = []

    const passage_id = props.original.passage_id;
    const query_id = props.original.query_id;
    if (this.state.activeQuestions[passage_id] === query_id) {
        const selectedAnswers = this.state.selectedAnswers[passage_id]
        if (selectedAnswers) {
            searchWords = selectedAnswers;
        }
    }
    return <WrapDiv><Highlighter highlightClassName='highlight-gold' searchWords={searchWords} textToHighlight={props.value} /></WrapDiv>
}
let renderAnswerCell = function(props) {
    const passage_id = props.original.passage_id;
    const query_id = props.original.query_id
    const activeQuestion = this.state.activeQuestions[passage_id]
    const selectedAnswers = this.state.selectedAnswers[passage_id]
    const searchWords = (activeQuestion === query_id) && selectedAnswers ? selectedAnswers : []
    return <WrapDiv><Highlighter highlightClassName='highlight-gold' searchWords={searchWords} textToHighlight={props.value} /></WrapDiv>
}

let selectedAnswerChange = function(rowInfo, e) {
    const passage_id = rowInfo.original.passage_id
    const query_id = rowInfo.original.query_id
    const answerDict = rowInfo.original.answer;
    const answerType = getAnswerField(answerDict)
    if (answerType && answerType.key !== 'date') {
        let selectedAnswer;
        selectedAnswer = (answerType.key === 'number') ? 
            [answerDict.number] : answerDict.spans
        
        this.setState({
            'activeQuestions': {
                ...this.state.activeQuestions,
                [passage_id]: query_id
            },
            'selectedAnswers': {
                ...this.state.selectedAnswers,
                [passage_id]: selectedAnswer
            }
        });
    }
}

function getAnswerForDisplay(raw_value) {
    let value = raw_value;
    if (Array.isArray(raw_value) && raw_value.length === 1) {
        value = value[0];
    } else if (typeof raw_value === 'object') {
        value = JSON.stringify(raw_value, null, 2);
    } else {
        value = parseInt(raw_value)
    }
    return value;
}

function getAnswerField(answerDict) {
    let answerType = null;

    const span_count = answerDict['spans'].length
    if (span_count > 0) {
        answerType = {key: 'spans', name: span_count > 1 ? 'Multi Span' : 'Single Span'}
    }
    else if (answerDict['number']) {
        answerType = {key: 'number', name: 'Number'};
    } else {
        const date = answerDict['date']
        if (date.day || date.month || date.year) {
            answerType = {key: 'date', name: 'Date'};
        }
    }

    return answerType;
}

function getAnswerType(answerDict) {
    let answerType = null;

    const span_count = answerDict['spans'].length
    if (span_count > 0) {
        if (span_count === 1) {
            answerType = 'single_span'
        } else {
            answerType = 'multi_span'
        }
    }
    else if (answerDict['number']) {
        answerType = 'number'
    } else {
        const date = answerDict['date']
        if (date.day || date.month || date.year) {
            answerType = 'date';
        }
    }
    return answerType;
}

function filterQAPairByAnswerType(qa_pair, filteredAnswerTypes) {
    const answerType = getAnswerType(qa_pair.answer)
    if (filteredAnswerTypes.includes(answerType)) {
        return true;
    }
    return false;
}

export default ExplorerTable;
