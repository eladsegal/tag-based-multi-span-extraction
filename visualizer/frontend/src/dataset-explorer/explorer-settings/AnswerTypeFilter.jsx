import React from 'react';
import {
    Input,
    FormGroup,
    Label
  } from 'reactstrap';

const answerTypes = [
    {'key': 'multi_span', 'value': 'Multi Span'},
    {'key': 'single_span', 'value': 'Single Span'},
    {'key': 'number', 'value': 'Number'},
    {'key': 'date', 'value': 'Date'}
];
class AnswerTypeFilter extends React.Component {
    constructor(props) {
        super(props);
        this.change = this.change.bind(this);
        this.state = { };
    }

    componentDidMount() {
        this.setState({ 
            filteredAnswerTypes: this.props.filteredAnswerTypes
        })
    }

    componentDidUpdate(prevProps, prevState) {
        if (prevState.filteredAnswerTypes !== this.state.filteredAnswerTypes) {
            this.props.onChange(this.state.filteredAnswerTypes);
        }
    }

    change(e) {
        const changedKey = e.currentTarget.getAttribute('answer-type-key');
        const changedKeyIndex = this.state.filteredAnswerTypes.indexOf(changedKey);
        if (changedKeyIndex !== -1) {
            const newfilteredAnswerTypes = [...this.state.filteredAnswerTypes];
            newfilteredAnswerTypes.splice(changedKeyIndex, 1);
            this.setState({ filteredAnswerTypes: newfilteredAnswerTypes });
        } else {
            this.setState({ filteredAnswerTypes: [...this.state.filteredAnswerTypes, changedKey] });
        }
    }

    render() { 
        return answerTypes.map(answerType => {
            return <FormGroup check key={answerType.key}>
                        <Label check>
                            <Input type="checkbox" 
                            onChange={this.change}
                            answer-type-key={answerType.key}
                            checked={(this.state.filteredAnswerTypes && 
                                this.state.filteredAnswerTypes.includes(answerType.key)) || 
                                false} 
                            />{answerType.value}
                        </Label>
                    </FormGroup>
        });
    }
}
 
export default AnswerTypeFilter;