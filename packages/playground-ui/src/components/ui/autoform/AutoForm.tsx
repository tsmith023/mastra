import React from 'react';
import { AutoForm as BaseAutoForm, AutoFormUIComponents } from '@autoform/react';
import { AutoFormProps } from './types';
import { Form } from './components/Form';
import { FieldWrapper } from './components/FieldWrapper';
import { ErrorMessage } from './components/ErrorMessage';
import { SubmitButton } from './components/SubmitButton';
import { StringField } from './components/StringField';
import { NumberField } from './components/NumberField';
import { BooleanField } from './components/BooleanField';
import { DateField } from './components/DateField';
import { SelectField } from './components/SelectField';
import { ObjectWrapper } from './components/ObjectWrapper';
import { ArrayWrapper } from './components/ArrayWrapper';
import { ArrayElementWrapper } from './components/ArrayElementWrapper';
import { RecordField } from './components/RecordField';

const ShadcnUIComponents: AutoFormUIComponents = {
  Form,
  FieldWrapper,
  ErrorMessage,
  SubmitButton,
  ObjectWrapper,
  ArrayWrapper,
  ArrayElementWrapper,
};

export const ShadcnAutoFormFieldComponents = {
  string: StringField,
  number: NumberField,
  boolean: BooleanField,
  date: DateField,
  select: SelectField,
  record: RecordField,
};
export type FieldTypes = keyof typeof ShadcnAutoFormFieldComponents;

export function AutoForm<T extends Record<string, any>>({
  uiComponents,
  formComponents,
  readOnly,
  ...props
}: AutoFormProps<T> & { readOnly?: boolean }) {
  return (
    <BaseAutoForm
      {...props}
      uiComponents={{ ...ShadcnUIComponents, ...uiComponents }}
      formComponents={{
        string: props => <StringField {...props} inputProps={{ ...props.inputProps, readOnly }} />,
        number: props => <NumberField {...props} inputProps={{ ...props.inputProps, readOnly }} />,
        boolean: props => <BooleanField {...props} inputProps={{ ...props.inputProps, readOnly }} />,
        date: props => <DateField {...props} inputProps={{ ...props.inputProps, readOnly }} />,
        select: props => <SelectField {...props} inputProps={{ ...props.inputProps, readOnly }} />,
        record: props => <RecordField {...props} inputProps={{ ...props.inputProps, readOnly }} />,
        ...formComponents,
      }}
    />
  );
}
